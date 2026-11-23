"""
Pre-computes the VAE latents of rendered OpenWebText training pages into a single file in --data-dir:

  train_latents.f16          raw float16 VAE latent samples, shape (N, 4, H/8, W/8), unscaled (same format as
                             validation_latents.f16)
  train_latents.json         metadata read by train.py (N, shape, rendering settings, ...)
  train_latents_progress/    one tiny JSON per task (resume point; train.py checks every task is done)

The file is filled by --num-tasks independent jobs (a slurm array, --task = $SLURM_ARRAY_TASK_ID): task k writes
pages [k*N/K, (k+1)*N/K) at its own offset in the preallocated file, so tasks need no coordination. Each task renders
whole row groups (1000 documents) of the training shards (0-78) in a fixed shuffled order, with the same per-document
augmentation seed as validation (f"{doc_id}-0"), and saves its progress after every row group, so a re-submitted
task resumes where it stopped. Pages of one document are stored consecutively; train.py samples pages at random.
"""
import argparse
import functools
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path
from time import time

import numpy as np
import pyarrow.parquet as pq
import torch
from diffusers.models import AutoencoderKL

from text_data import VAL_SHARD, owt_documents, owt_files, render_document
from text_eval import encode


def render_pages(job, **render_kwargs):
    doc_id, text = job
    random.seed(f"{doc_id}-0")
    return [np.array(img) for img, _ in render_document(text, **render_kwargs)]  # numpy: cheap to pickle


def write_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic: a killed job never leaves a half-written file


@torch.no_grad()
def main(args):
    assert 0 <= args.task < args.num_tasks
    latent_size = args.image_size // 8
    shape = [4, latent_size, latent_size]
    page_bytes = 2 * int(np.prod(shape))  # float16
    num_pages = int(args.size_gb * 1e9) // page_bytes
    start, end = args.task * num_pages // args.num_tasks, (args.task + 1) * num_pages // args.num_tasks

    os.makedirs(f"{args.data_dir}/train_latents_progress", exist_ok=True)
    meta = {"num_pages": num_pages, "shape": shape, "dtype": "float16", "scaled": False, "vae": args.vae,
            "image_size": args.image_size, "font_size": args.font_size, "seed": args.seed,
            "num_tasks": args.num_tasks}
    meta_path = f"{args.data_dir}/train_latents.json"
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        assert old == meta, f"{meta_path} was written with different settings:\n{old}\nvs now:\n{meta}"
    write_json(meta, meta_path)

    # Preallocate the whole file (sparse); concurrent tasks may all do this safely (same size, no data lost):
    path = f"{args.data_dir}/train_latents.f16"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size < num_pages * page_bytes:
        os.ftruncate(fd, num_pages * page_bytes)
    assert os.fstat(fd).st_size == num_pages * page_bytes, f"{path} has the wrong size (different --size-gb?)"
    f = os.fdopen(fd, "r+b")

    progress_path = f"{args.data_dir}/train_latents_progress/{args.task:03d}.json"
    progress = json.load(open(progress_path)) if os.path.exists(progress_path) else {"units_done": 0, "pages": 0}
    progress.update(start=start, end=end)
    if progress.get("done"):
        print(f"Task {args.task} already done ({end - start:,} pages)")
        return

    # Row groups of the training shards in a fixed shuffled order, split between tasks:
    units = [(file, rg) for file in owt_files() if f"-{VAL_SHARD:05d}-" not in Path(file).name
             for rg in range(pq.ParquetFile(file).metadata.num_row_groups)]
    random.Random(args.seed).shuffle(units)
    units = units[args.task::args.num_tasks]

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to("cuda").eval()
    vae = vae.to(memory_format=torch.channels_last)
    render = functools.partial(render_pages, size=args.image_size, font_size=args.font_size)
    pages = progress["pages"]
    print(f"Task {args.task}/{args.num_tasks}: pages [{start:,}, {end:,}) of {num_pages:,} "
          f"({num_pages * page_bytes / 1e9:.1f} GB total), resuming at {pages:,} pages / "
          f"row group {progress['units_done']} of {len(units)}")

    def write(batch):
        nonlocal pages
        n = min(len(batch), end - start - pages)
        if n <= 0:
            return
        x = torch.from_numpy(np.stack(batch[:n])).cuda().permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
        z = encode(vae, x, args.encode_chunk)
        f.seek((start + pages) * page_bytes)
        f.write(z.to(torch.float16).cpu().numpy().tobytes())
        pages += n

    t0, pages0 = time(), pages
    with Pool(args.workers) as pool:
        # One row group is rendered ahead while the current one is encoded (bounded memory, no idle workers):
        def submit(i):
            file, rg = units[i]
            return pool.imap(render, owt_documents(file, [rg]), chunksize=8)

        i = progress["units_done"]
        nxt = submit(i)
        while pages < end - start:
            assert i < len(units), f"task {args.task} ran out of text ({pages:,} of {end - start:,} pages)"
            cur, nxt = nxt, (submit(i + 1) if i + 1 < len(units) else None)
            buffer = []
            for imgs in cur:
                buffer.extend(imgs)
                while len(buffer) >= args.batch_size:
                    write(buffer[:args.batch_size])
                    buffer = buffer[args.batch_size:]
            write(buffer)
            f.flush()
            os.fsync(f.fileno())  # data on disk before the progress that points past it
            i += 1
            progress.update(units_done=i, pages=pages, done=pages >= end - start)
            write_json(progress, progress_path)
            rate = (pages - pages0) / (time() - t0)
            print(f"row group {i}: {pages:,}/{end - start:,} pages, {rate:.0f} pages/s, "
                  f"ETA {(end - start - pages) / rate / 3600:.1f} h", flush=True)
    f.close()
    print(f"Task {args.task} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--size-gb", type=float, default=200, help="size of train_latents.f16 (decimal GB)")
    parser.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--font-size", type=int, default=35)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-chunk", type=int, default=128, help="VAE encode batch size (memory)")
    parser.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
