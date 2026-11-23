#!/bin/bash -l
#SBATCH --account=ecs
#SBATCH --partition=a100,swarm_a100
#SBATCH --job-name=text_sit_prepare
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# Renders the held-out OpenWebText validation pages and computes the evaluation references (~10k pages, small).
# Training latents come from text_sit_latents.slurm (can run at the same time). Submit all three jobs with:
#   a=$(sbatch --parsable text_sit_prepare.slurm) && b=$(sbatch --parsable text_sit_latents.slurm) && \
#     sbatch --dependency=afterok:$a:$b text_sit_train.slurm

echo Script starting
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
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python prepare_eval_data.py --data-dir $TEXT_SIT_ROOT/data_256 --image-size 256 --font-size 35 --workers 15 || exit 1

echo Script completed =3
