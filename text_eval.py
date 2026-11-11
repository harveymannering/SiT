"""
Text-quality (OCR -> LLM perplexity) and FID metrics for rendered-text images.
"""
import math
import re

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from torchmetrics.image.fid import FrechetInceptionDistance

FID_STATES = ["real_features_sum", "real_features_cov_sum", "real_features_num_samples"]


def create_fid(device, ref_path=None):
    """FID metric; real-image statistics are loaded from ref_path (written by prepare_text_latents.py)."""
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False)
    if ref_path is not None:
        for k, v in torch.load(ref_path, map_location="cpu").items():
            setattr(fid, k, v)
    return fid.to(device)


def save_fid_ref(fid, path):
    torch.save({k: getattr(fid, k) for k in FID_STATES}, path)


@torch.no_grad()
def encode(vae, x, chunk_size=64):
    """VAE-encodes [-1, 1] images in chunks with bf16 autocast; returns unscaled latent samples (float32)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = [vae.encode(c.contiguous(memory_format=torch.channels_last)).latent_dist.sample() for c in x.split(chunk_size)]
    return torch.cat(z).float()


@torch.no_grad()
def decode(vae, z, chunk_size=16):
    """VAE-decodes unscaled latents in chunks (a 512px decode needs a lot of activation memory)."""
    return torch.cat([vae.decode(c).sample for c in z.split(chunk_size)])


def to_uint8(x):
    """[-1, 1] float (N, C, H, W) -> uint8 (N, C, H, W)."""
    return x.clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)


class TextQuality:
    """OCR (docTR, word-level recogniser so it has little language prior to 'fix' garbled text),
    then perplexity of the recognised text under a pretrained LLM."""
    def __init__(self, device, llm="Qwen/Qwen3-1.7B-Base", vocab_path=None):
        from doctr.models import ocr_predictor
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.ocr = ocr_predictor(pretrained=True).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(llm, padding_side="right")
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(llm, torch_dtype=torch.bfloat16).to(device).eval()
        self.vocab = set(open(vocab_path).read().split()) if vocab_path else None

    @staticmethod
    def _reading_order(page):
        # docTR's block grouping can interleave lines, so sort words into lines by vertical centre, then by x
        words = sorted((w for b in page.blocks for l in b.lines for w in l.words),
                       key=lambda w: w.geometry[0][1] + w.geometry[1][1])
        lines = []
        for w in words:
            yc, h = (w.geometry[0][1] + w.geometry[1][1]) / 2, w.geometry[1][1] - w.geometry[0][1]
            if lines and abs(yc - lines[-1][-1][0]) < h / 2:
                lines[-1].append((yc, w))
            else:
                lines.append([(yc, w)])
        return " ".join(w.value for l in lines for _, w in sorted(l, key=lambda t: t[1].geometry[0][0]))

    @torch.no_grad()
    def read(self, images, batch_size=32):
        """images: uint8 tensor (N, 3, H, W) or list of HWC uint8 arrays -> list of strings."""
        if torch.is_tensor(images):
            images = list(images.permute(0, 2, 3, 1).cpu().numpy())
        texts = []
        for i in range(0, len(images), batch_size):
            texts += [self._reading_order(p) for p in self.ocr(images[i:i + batch_size]).pages]
        return texts

    @torch.no_grad()
    def perplexity(self, texts, batch_size=8):
        """Corpus-level perplexity: exp(total NLL / total predicted tokens). Empty texts are skipped."""
        texts = [t for t in texts if t.strip()]
        nll, n = 0.0, 0
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], return_tensors="pt", padding=True).to(self.device)
            logits = self.lm(**enc).logits[:, :-1].float()
            labels = enc.input_ids[:, 1:].masked_fill(enc.attention_mask[:, 1:] == 0, -100)
            nll += torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="sum").item()
            n += (labels != -100).sum().item()
        return math.exp(nll / n) if n else float("nan")

    def __call__(self, images, refs=None):
        texts = self.read(images)
        words = [w for t in texts for w in re.findall(r"[a-z]+", t.lower())]
        metrics = {
            "ppl": self.perplexity(texts),
            "ocr_words_per_image": np.mean([len(t.split()) for t in texts]),
        }
        if self.vocab is not None:  # fraction of recognised words that are real (wikitext-103 train) words
            metrics["valid_word_frac"] = np.mean([w in self.vocab for w in words]) if words else float("nan")
        if refs is not None:  # character error rate vs. the rendered text (whitespace ignored)
            norm = lambda s: "".join(s.split())
            metrics["cer"] = np.mean([Levenshtein.normalized_distance(norm(t), norm(r)) for t, r in zip(texts, refs)])
        return metrics, texts
