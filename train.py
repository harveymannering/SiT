# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from datetime import timedelta
from glob import glob
from time import time
import argparse
import json
import logging
import os

from models import SiT_models
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
from text_data import RenderedOpenWebText, owt_files
from text_eval import TextQuality, create_fid, decode, encode, to_uint8
import wandb_utils


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def measure_flops(model, transport, x, model_kwargs, bf16):
    """FLOPs of one training step (forward + backward), counted by PyTorch."""
    with FlopCounterMode(display=False) as counter:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss = transport.training_losses(model, x, model_kwargs)["loss"].mean()
        loss.backward()
    model.zero_grad(set_to_none=True)
    return counter.get_total_flops()


@torch.no_grad()
def validation_loss(model, transport, val_latents, batch_size, bf16, seed=0):
    """Loss on the whole validation set with fixed noise and timesteps, so it is comparable across steps and models."""
    was_training = model.training
    model.eval()  # no label dropout
    losses = []
    with torch.random.fork_rng(devices=[val_latents.device]):
        torch.manual_seed(seed)
        for x in val_latents.split(batch_size):
            x = x.float().mul(0.18215)
            y = torch.zeros(len(x), dtype=torch.long, device=x.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                losses.append(transport.training_losses(model, x, dict(y=y))["loss"].float())
    model.train(was_training)
    return torch.cat(losses).mean().item()


@torch.no_grad()
def evaluate(ema, vae, transport_sampler, zs, fid, text_quality, args):
    """Generates len(zs) samples with the EMA model -> FID, and OCR/perplexity on the first args.num_ocr_samples."""
    sample_fn = transport_sampler.sample_ode(sampling_method=args.sampling_method, num_steps=args.num_sampling_steps)

    def model_fn(x, t, y):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            return ema(x, t, y).float()

    images = []
    for z in zs.split(args.sample_batch_size):
        y = torch.zeros(len(z), dtype=torch.long, device=z.device)
        samples = sample_fn(z, model_fn, y=y)[-1]
        images.append(to_uint8(decode(vae, samples / 0.18215, args.decode_chunk)))
    images = torch.cat(images)
    fid.reset()  # keeps the real-image statistics
    for x in images.split(256):
        fid.update(x, real=False)
    metrics, texts = text_quality(images[:args.num_ocr_samples])
    metrics["fid"] = fid.compute().item()
    return metrics, images, texts


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl", timeout=timedelta(hours=2))  # long: other ranks wait while rank 0 evaluates
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder (a fixed --exp-name lets a re-submitted job resume from its latest checkpoint):
    model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
    experiment_name = args.exp_name or model_string_name
    experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    sample_dir = f"{experiment_dir}/samples"  # Stores sample grids + OCR text from each evaluation
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    def log_metrics(metrics, step):
        """Appends to metrics.jsonl (the main record of the run) and wandb."""
        if rank == 0:
            metrics = {"step": step, "flops": step * flops_per_step, "images_seen": step * args.global_batch_size,
                       "wall_time": time() - run_start, **metrics}
            with open(f"{experiment_dir}/metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")
            if args.wandb:
                wandb_utils.log(metrics, step=step)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = model.to(device)

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity;
    transport_sampler = Sampler(transport)

    # FLOPs per training step (all GPUs; SiT forward + backward only, since latents are pre-computed):
    x = torch.randn(local_batch_size, 4, latent_size, latent_size, device=device)
    y = torch.zeros(local_batch_size, dtype=torch.long, device=device)
    flops_per_step = measure_flops(model, transport, x, dict(y=y), args.bf16) * dist.get_world_size()
    logger.info(f"Training FLOPs per step: {flops_per_step:.3e}")

    model = DDP(model, device_ids=[device])
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device, memory_format=torch.channels_last)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Resume from --ckpt, or else from the latest checkpoint in this experiment's folder:
    train_steps = 0
    ckpts = sorted(glob(f"{checkpoint_dir}/*.pt"))
    if args.ckpt is None and ckpts:
        args.ckpt = ckpts[-1]
    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.module.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        opt.load_state_dict(state_dict["opt"])
        train_steps = state_dict.get("train_steps", 0)
        logger.info(f"Resumed from {args.ckpt} at step {train_steps}")

    # Setup data: OpenWebText pages rendered on the fly (text_data.py); validation latents from prepare_eval_data.py
    dataset = RenderedOpenWebText(
        owt_files(),
        rank=rank,
        world_size=dist.get_world_size(),
        seed=args.global_seed,
        pages_seen=train_steps * local_batch_size,  # resume without repeating pages
        size=args.image_size,
        font_size=args.font_size
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    val_latents = np.fromfile(f"{args.data_path}/validation_latents.f16", dtype=np.float16)
    val_latents = torch.from_numpy(val_latents.reshape(-1, 4, latent_size, latent_size)[:args.num_val]).to(device)
    logger.info(f"Training on {len(dataset.units):,} OpenWebText row groups (1000 documents each), "
                f"validating on {len(val_latents):,} held-out pages")

    # Prepare models for training:
    if train_steps == 0:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = run_start = time()

    # Evaluation (rank 0 only): same noise for every evaluation and every model size, real-image FID statistics
    # and the OCR + LLM text-quality model. Unconditional: every image has label 0.
    if rank == 0:
        zs = torch.randn(args.num_fid_samples, 4, latent_size, latent_size,
                         generator=torch.Generator().manual_seed(args.global_seed)).to(device)
        fid = create_fid(device, f"{args.data_path}/fid_ref.pt")
        text_quality = TextQuality(device, args.ppl_model, f"{args.data_path}/vocab.txt")
    # Evaluate every --eval-every steps and also at FLOP budgets eval_flops_start * eval_flops_factor^k, so all model
    # sizes are evaluated at the same compute (log-spaced, for scaling plots):
    next_eval_flops = args.eval_flops_start
    while next_eval_flops <= train_steps * flops_per_step:
        next_eval_flops *= args.eval_flops_factor

    def run_eval(trigger):
        if rank == 0:
            logger.info(f"Evaluating ({trigger})...")
            eval_start = time()
            metrics, images, texts = evaluate(ema, vae, transport_sampler, zs, fid, text_quality, args)
            metrics["val_loss_ema"] = validation_loss(ema, transport, val_latents, args.sample_batch_size, args.bf16)
            metrics = {f"eval/{k}": v for k, v in metrics.items()}
            metrics.update({"eval/trigger": trigger, "eval/time": time() - eval_start})
            log_metrics(metrics, train_steps)
            logger.info(f"(step={train_steps:07d}) " + ", ".join(f"{k}: {v}" for k, v in metrics.items()))
            save_image(images[:args.num_saved_samples].float() / 255, f"{sample_dir}/{train_steps:07d}.png",
                       nrow=int(args.num_saved_samples ** 0.5))
            with open(f"{sample_dir}/{train_steps:07d}_ocr.jsonl", "w") as f:
                f.writelines(json.dumps(t) + "\n" for t in texts)
            if args.wandb:
                wandb_utils.log_image(images[:args.num_saved_samples].float() / 127.5 - 1, train_steps)
        dist.barrier()

    logger.info(f"Training for {args.max_steps} steps...")
    while train_steps < args.max_steps:
        for x, y in loader:  # endless stream
            x = x.to(device, non_blocking=True).float().div_(127.5).sub_(1)  # uint8 -> [-1, 1]
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = encode(vae, x, args.encode_chunk).mul_(0.18215)
            model_kwargs = dict(y=y)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.detach()  # no .item() here: avoids a GPU sync every step
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = running_loss / log_steps
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                log_metrics({"train_loss": avg_loss, "train_steps_per_sec": steps_per_sec,
                             "train_tflops_per_sec": steps_per_sec * flops_per_step / 1e12}, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Validation loss (cheap: fixed noise/timesteps, whole validation set):
            if train_steps % args.val_every == 0 and rank == 0:
                log_metrics({"val_loss": validation_loss(model.module, transport, val_latents,
                                                         args.sample_batch_size, args.bf16)}, train_steps)

            # Save SiT checkpoint:
            if (train_steps % args.ckpt_every == 0 and train_steps > 0) or train_steps == args.max_steps:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            # Evaluate (FID, OCR -> perplexity, EMA validation loss):
            trigger = []
            if train_steps % args.eval_every == 0 or train_steps == args.max_steps:
                trigger.append("steps")
            if train_steps * flops_per_step >= next_eval_flops:
                trigger.append("flops")
                while next_eval_flops <= train_steps * flops_per_step:
                    next_eval_flops *= args.eval_flops_factor
            if trigger:
                run_eval("+".join(trigger))
                start_time = time()  # don't count evaluation time in steps/sec

            if train_steps >= args.max_steps:
                break

    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1)  # unconditional: every image has label 0
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--exp-name", type=str, default=None, help="experiment folder name (default: model name)")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    # Evaluation:
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--num-val", type=int, default=4096, help="validation pages used for the val loss")
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-flops-start", type=float, default=1e16)
    parser.add_argument("--eval-flops-factor", type=float, default=2.0)
    parser.add_argument("--num-fid-samples", type=int, default=2048)
    parser.add_argument("--num-ocr-samples", type=int, default=512)
    parser.add_argument("--num-saved-samples", type=int, default=16)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--decode-chunk", type=int, default=16, help="VAE decode batch size (memory)")
    parser.add_argument("--encode-chunk", type=int, default=32, help="VAE encode batch size (memory)")
    parser.add_argument("--font-size", type=int, default=40)
    parser.add_argument("--sampling-method", type=str, default="euler")
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--ppl-model", type=str, default="Qwen/Qwen3-1.7B-Base")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
