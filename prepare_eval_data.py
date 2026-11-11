"""
Prepares the (small, fixed) evaluation data in --data-dir. Training pages are rendered on the fly (text_data.py).

  vocab.txt                  words seen >= 3 times in one OpenWebText training shard (for the valid-word metric)
  validation_latents.f16     raw float16 VAE latents of the validation pages, shape (N, 4, 64, 64), unscaled
  validation_texts.jsonl     the text drawn on each validation page
  fid_ref.pt                 Inception statistics of the real validation pages
  reference.json             metrics on real and VAE-reconstructed validation pages (sanity check / ceiling)

Validation pages come from the first documents of the held-out OpenWebText shard (never used for training), each
rendered with the same per-document seed as training uses, so the output is identical on every run.
"""
import argparse
import json
import os
import random
import re
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch
from diffusers.models import AutoencoderKL

from text_data import VAL_SHARD, owt_documents, owt_files, render_document
from text_eval import TextQuality, create_fid, decode, encode, save_fid_ref, to_uint8


def render_pages(job):
    doc_id, text = job
    random.seed(f"{doc_id}-0")
    return [(np.array(img), page_text) for img, page_text in render_document(text)]  # numpy: cheap to pickle


def validation_pages(args):
    n = 0
    with Pool(args.workers) as pool:
        for result in pool.imap(render_pages, owt_documents(owt_files()[VAL_SHARD]), chunksize=4):  # keeps order
            for page in result:
                if n >= args.num_val:
                    return
                yield page
                n += 1


def batches(it, size):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


@torch.no_grad()
def main(args):
    os.makedirs(args.data_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    vocab_path = os.path.join(args.data_dir, "vocab.txt")
    if not os.path.exists(vocab_path):
        counts = Counter(w for _, t in owt_documents(owt_files()[0]) for w in re.findall(r"[a-z]+", t.lower()))
        open(vocab_path, "w").write("\n".join(w for w, c in counts.items() if c >= 3))

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to("cuda").eval()
    fid = create_fid("cuda")
    real, recon, refs = [], [], []
    with open(os.path.join(args.data_dir, "validation_latents.f16"), "wb") as f, \
         open(os.path.join(args.data_dir, "validation_texts.jsonl"), "w") as texts:
        for batch in batches(validation_pages(args), args.batch_size):
            x = torch.from_numpy(np.stack([img for img, _ in batch])).permute(0, 3, 1, 2).cuda()
            z = encode(vae, x.float().div(127.5).sub(1), args.batch_size)
            f.write(z.to(torch.float16).cpu().numpy().tobytes())
            texts.writelines(json.dumps(t) + "\n" for _, t in batch)
            fid.update(x, real=True)
            if len(real) < args.num_ref:  # keep some real + VAE-reconstructed pages for the reference metrics
                k = args.num_ref - len(real)
                real.extend(x[:k].cpu())
                recon.extend(to_uint8(decode(vae, z[:k], args.decode_chunk)).cpu())
                refs.extend(t for _, t in batch[:k])
    print(f"Wrote {fid.real_features_num_samples.item():,} validation pages")
    save_fid_ref(fid, os.path.join(args.data_dir, "fid_ref.pt"))

    fid_recon = create_fid("cuda", os.path.join(args.data_dir, "fid_ref.pt"))  # FID floor set by the VAE
    fid_recon.update(torch.stack(recon).cuda(), real=False)
    reference = {"num_ref": len(real), "fid_vae_recon": fid_recon.compute().item()}
    vae.cpu()  # free GPU memory for the OCR + LLM
    quality = TextQuality("cuda", args.ppl_model, vocab_path)
    for name, imgs in [("real", real), ("vae_recon", recon)]:
        metrics, _ = quality(torch.stack(imgs), refs)
        reference.update({f"{name}_{k}": v for k, v in metrics.items()})
    reference["gt_text_ppl"] = quality.perplexity(refs)  # PPL of the ground-truth text, no OCR
    json.dump(reference, open(os.path.join(args.data_dir, "reference.json"), "w"), indent=2)
    print("Reference metrics:", json.dumps(reference, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--ppl-model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--num-val", type=int, default=10_000, help="validation pages (val loss + FID statistics)")
    parser.add_argument("--num-ref", type=int, default=2048, help="validation pages used for the reference metrics")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decode-chunk", type=int, default=16)
    parser.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
