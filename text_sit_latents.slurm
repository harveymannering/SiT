#!/bin/bash -l
#SBATCH --account=ecs
#SBATCH --partition=l40,a100,swarm_a100,swarm_h100
#SBATCH --job-name=text_sit_latents
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-7

# Renders + VAE-encodes 200 GB of OpenWebText training pages (~24.4M pages, 8 KB each) into ONE file,
# $TEXT_SIT_ROOT/data_256/train_latents.f16. Each array task fills its own 1/8 of the file (25 GB, ~3M pages),
# so NUM_TASKS must stay equal to the array size. Re-submitting resumes unfinished tasks (finished ones exit at once).
# --image-size / --font-size must match text_sit_prepare.slurm.
NUM_TASKS=8

echo Script starting: task $SLURM_ARRAY_TASK_ID of $NUM_TASKS
nvidia-smi

# Compute nodes are offline: run `source cluster_env.sh && python prefetch.py` on the login node first.
source cluster_env.sh
conda activate $TEXT_SIT_ENV
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python prepare_train_latents.py \
    --data-dir $TEXT_SIT_ROOT/data_256 \
    --size-gb 200 \
    --task $SLURM_ARRAY_TASK_ID \
    --num-tasks $NUM_TASKS \
    --image-size 256 \
    --font-size 35 \
    --vae mse \
    --workers 15 || exit 1

echo Script completed =3
