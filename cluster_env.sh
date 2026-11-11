# Shared by the slurm scripts and the login-node prefetch step, so both use the same caches.
# Usage on the login node:  source cluster_env.sh && python prefetch.py
export TEXT_SIT_ROOT=/scratch/hgm1g14/text_sit           # generated data (eval references) + results
export TEXT_SIT_ENV=/scratch/hgm1g14/conda_envs/sit      # conda env (see environment.yml)

# Every download (models + data) goes under cache_dir:
CACHE_DIR=/scratch/hgm1g14/cache_dir
export HF_HOME=$CACHE_DIR/huggingface                    # OpenWebText parquet, VAE, Qwen
export HF_HUB_CACHE=$HF_HOME/hub                         # explicit: overrides any value set in ~/.bashrc
unset TRANSFORMERS_CACHE                                 # legacy variable; would take priority over HF_HOME
export DOCTR_CACHE_DIR=$CACHE_DIR/doctr                  # OCR weights
export TORCH_HOME=$CACHE_DIR/torch                       # FID Inception weights
export FONT_DIR=$CACHE_DIR/fonts                         # Google Fonts used for rendering
export PIP_CACHE_DIR=$CACHE_DIR/pip                      # in case you pip install with this sourced
