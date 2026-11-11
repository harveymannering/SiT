# Shared by the slurm scripts and the login-node prefetch step, so both use the same caches.
# Usage on the login node:  source cluster_env.sh && python prefetch.py
export TEXT_SIT_ROOT=/scratch/hgm1g14/text_sit           # data + results
export TEXT_SIT_ENV=/scratch/hgm1g14/conda_envs/sit      # conda env (see environment.yml)
export HF_HOME=/scratch/hgm1g14/cache_dir/huggingface    # wikitext, VAE, Qwen
export DOCTR_CACHE_DIR=/scratch/hgm1g14/cache_dir/doctr  # OCR weights
export TORCH_HOME=/scratch/hgm1g14/cache_dir/torch       # FID Inception weights
