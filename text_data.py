"""
Rendered-text dataset: OpenWebText documents rendered on the fly (same rendering/augmentation as render_wikitext.ipynb).
"""
import random
import textwrap
import urllib.request
from glob import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw, ImageFont

# Google Fonts (OFL), downloaded once into ./fonts (run prefetch.py on a node with internet)
FONT_DIR = Path(__file__).parent / "fonts"
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/"
FONT_FILES = ["lato/Lato-Regular.ttf", "roboto/Roboto[wdth,wght].ttf", "opensans/OpenSans[wdth,wght].ttf",
              "montserrat/Montserrat[wght].ttf", "oswald/Oswald[wght].ttf", "playfairdisplay/PlayfairDisplay[wght].ttf",
              "sourceserif4/SourceSerif4[opsz,wght].ttf", "ebgaramond/EBGaramond[wght].ttf", "ptserif/PT_Serif-Web-Regular.ttf",
              "merriweather/Merriweather[opsz,wdth,wght].ttf", "courierprime/CourierPrime-Regular.ttf", "chivo/Chivo[wght].ttf"]

# OpenWebText: 80 parquet shards of ~100k documents (row groups of 1000). The last shard is held out for validation.
OWT_REPO = "Skylion007/openwebtext"
VAL_SHARD = 79
MAX_CHARS_PER_PAGE = 400  # upper bound (measured max ~305), used to skip already-seen documents when resuming


def get_fonts():
    fonts = []
    for f in FONT_FILES:
        p = FONT_DIR / Path(f).name
        if not p.exists():
            p.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(FONT_URL + f.replace("[", "%5B").replace("]", "%5D"), p)
        fonts.append(str(p))
    return fonts


FONTS = get_fonts()


def render(text, size=512, font_size=40, pad=8, font=None):
    """Returns (image, number of words drawn). Lines that don't fit at the bottom are dropped."""
    text = text.strip()
    font = ImageFont.truetype(font or random.choice(FONTS), font_size)
    if "[" in font.path: font.set_variation_by_name(b"Regular")  # variable fonts: some default to Thin
    bg = tuple(random.randint(230, 255) for _ in range(3))  # white / off-white / light grey / beige tints
    fg = tuple(random.randint(0, 60) for _ in range(3))     # black / dark grey / very dark blue, red, ...
    sample = text[:500]
    chars = int(0.9 * (size - 2 * pad) * len(sample) / font.getlength(sample))  # chars/line from mean glyph width
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    max_lines = (size - 2 * pad) // (font.getbbox("A")[3] + 4) + 1  # upper bound (PIL line spacing), for speed
    lines = textwrap.wrap(text[:chars * (max_lines + 2)], chars, break_long_words=False, break_on_hyphens=False)
    lines = lines[:max_lines]  # whole words only (text was truncated mid-word well after max_lines)
    while lines and draw.multiline_textbbox((pad, pad), "\n".join(lines), font=font)[3] > size - pad:
        lines.pop()  # drop lines that would be cut off at the bottom
    draw.multiline_text((pad, pad), "\n".join(lines), font=font, fill=fg)
    return img, sum(len(l.split()) for l in lines)


def render_document(document, **kwargs):
    """Renders a whole document as consecutive pages (no text is repeated or skipped; newlines become spaces).
    Yields (image, text drawn on that page); a trailing page with <= 50 chars is dropped."""
    words = document.split()
    while len(" ".join(words)) > 50:
        img, n = render(" ".join(words), **kwargs)
        if n == 0:
            break
        yield img, " ".join(words[:n])
        words = words[n:]


def owt_files():
    """Local paths of the OpenWebText parquet shards (downloaded by prefetch.py; offline afterwards)."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(OWT_REPO, repo_type="dataset", allow_patterns="plain_text/*.parquet")
    files = sorted(glob(f"{path}/plain_text/*.parquet"))
    assert len(files) == 80, f"expected 80 OpenWebText shards, found {len(files)} in {path}"
    return files


def owt_documents(file, row_groups=None):
    """Yields (document id, text); the id seeds the rendering augmentation, so every run renders a document the same."""
    f = pq.ParquetFile(file)
    shard = int(Path(file).name.split("-")[1])
    for rg in row_groups if row_groups is not None else range(f.num_row_groups):
        for i, text in enumerate(f.read_row_group(rg, columns=["text"]).column("text").to_pylist()):
            yield f"{shard}-{rg}-{i}", text


def to_uint8_tensor(img):
    """PIL -> uint8 tensor (C, H, W)."""
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)


class RenderedOpenWebText(torch.utils.data.IterableDataset):
    """Endless stream of rendered OpenWebText training pages (shards 0-78), rendered on the fly. Yields (uint8 image, 0).

    Row groups (1000 documents each) are shuffled with a fixed seed and split between all (rank, worker) pairs; each
    document is rendered into consecutive pages, which a shuffle buffer mixes with other documents. One pass is ~140M
    unique pages. To resume, pass pages_seen (pages this rank has trained on): each worker skips at least as much text
    as it could have rendered, so no page is repeated (a few unseen documents are skipped instead)."""
    def __init__(self, files, rank, world_size, seed=0, pages_seen=0, buffer_size=1024, **render_kwargs):
        self.units = [(f, rg) for f in files if f"-{VAL_SHARD:05d}-" not in Path(f).name
                      for rg in range(pq.ParquetFile(f).metadata.num_row_groups)]
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.pages_seen, self.buffer_size, self.render_kwargs = pages_seen, buffer_size, render_kwargs

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        workers, worker_id = (info.num_workers, info.id) if info else (1, 0)
        shard, num_shards = self.rank * workers + worker_id, self.world_size * workers
        rng = random.Random(f"{self.seed}-{shard}")
        # Each worker supplies an equal share of this rank's batches; also skip its shuffle buffer + prefetched pages
        skip_chars = (self.pages_seen // workers + self.buffer_size + 1024) * MAX_CHARS_PER_PAGE if self.pages_seen else 0
        buffer, epoch = [], 0
        while True:
            units = self.units[:]
            random.Random(f"{self.seed}-epoch{epoch}").shuffle(units)
            for file, rg in units[shard::num_shards]:
                for doc_id, text in owt_documents(file, [rg]):
                    if skip_chars > 0:
                        skip_chars -= len(text)
                        continue
                    random.seed(f"{doc_id}-{epoch}")  # deterministic augmentation per document
                    for img, _ in render_document(text, **self.render_kwargs):
                        buffer.append(to_uint8_tensor(img))
                        if len(buffer) >= self.buffer_size:
                            yield buffer.pop(rng.randrange(len(buffer))), 0
            epoch += 1
