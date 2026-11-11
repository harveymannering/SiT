#!/bin/bash -l
#SBATCH --account=ecs
#SBATCH --partition=dual_h200,quad_h200
#SBATCH --job-name=text_sit_prepare
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# Renders the held-out OpenWebText validation pages and computes the evaluation references (~10k pages, small).
# Training pages are rendered on the fly. Submit both jobs with:
#   jid=$(sbatch --parsable text_sit_prepare.slurm) && sbatch --dependency=afterok:$jid text_sit_train.slurm

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
activate_or_die $TEXT_SIT_ENV
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python prepare_eval_data.py --data-dir $TEXT_SIT_ROOT/data --workers 15 || exit 1

echo Script completed =3
