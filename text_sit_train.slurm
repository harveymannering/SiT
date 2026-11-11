#!/bin/bash -l
#SBATCH --account=ecs
#SBATCH --partition=dual_h200,quad_h200
#SBATCH --job-name=text_sit_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=60:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4

# Model-size sweep: one array task per model (~32M -> 453M params, roughly x2 per step).
# OpenWebText pages are rendered (CPU workers) and VAE-encoded on the fly, so every training page is unique.
# Needs text_sit_prepare.slurm to have finished. Re-submitting resumes each model from its latest checkpoint.
MODELS=(SiT-S/8 SiT-M/8 SiT-B/8 SiT-BL/8 SiT-L/8)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo Script starting: $MODEL
nvidia-smi

# `conda activate` can return 0 even when the env doesn't exist, so check $CONDA_PREFIX instead.
activate_or_die() {
    conda activate "$1"
    if [[ "$CONDA_PREFIX" != "$1" ]]; then
        echo "ERROR: failed to activate conda env $1 (still in ${CONDA_PREFIX:-<none>})"
        exit 1
    fi
}

# Compute nodes are offline: run `source cluster_env.sh && python prefetch.py` on the login node first.
source cluster_env.sh
activate_or_die $TEXT_SIT_ENV
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Metrics go to $TEXT_SIT_ROOT/results/<model>/metrics.jsonl (train/val loss, FID, OCR perplexity, FLOPs),
# with sample grids + OCR text in .../samples/.
torchrun --standalone --nproc_per_node=1 train.py \
    --data-path $TEXT_SIT_ROOT/data \
    --results-dir $TEXT_SIT_ROOT/results \
    --exp-name ${MODEL//\//-} \
    --model $MODEL \
    --image-size 512 \
    --vae mse \
    --global-batch-size 256 \
    --max-steps 200000 \
    --num-workers 15 \
    --ckpt-every 50000 \
    --eval-every 20000 || exit 1

echo Script completed =3
