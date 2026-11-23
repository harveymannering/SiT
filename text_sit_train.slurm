#!/bin/bash -l
#SBATCH --account=ecs
#SBATCH --partition=l40,a100,swarm_a100,swarm_h100
#SBATCH --job-name=text_sit_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G                 # 200 GB of training latents are held in RAM
#SBATCH --time=60:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-6

# Model-size sweep over patch sizes 8 and 4: one array task per model (~32M -> ~675M params, roughly x2 per step).
# Trains on the 200 GB of pre-computed latents (~24M OpenWebText pages) from text_sit_latents.slurm.
# Needs text_sit_prepare.slurm and text_sit_latents.slurm to have finished. Re-submitting resumes each model from its latest checkpoint.
MODELS=(SiT-S/8 SiT-M/8 SiT-B/8 SiT-BL/8 SiT-L/8 SiT-XL/8 SiT-XL/4)
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
conda activate $TEXT_SIT_ENV
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Metrics go to $TEXT_SIT_ROOT/results/<model>/metrics.jsonl (train/val loss, FID, OCR perplexity, FLOPs),
# with sample grids + OCR text in .../samples/.
torchrun --standalone --nproc_per_node=1 train.py \
    --data-path $TEXT_SIT_ROOT/data_256 \
    --results-dir $TEXT_SIT_ROOT/results_256 \
    --exp-name ${MODEL//\//-} \
    --model $MODEL \
    --image-size 256 \
    --vae mse \
    --global-batch-size 256 \
    --max-steps 200000 \
    --ckpt-every 50000 \
    --eval-every 20000 || exit 1

echo Script completed =3
