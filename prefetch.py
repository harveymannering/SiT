"""
Run once on the login node (internet access) after `source cluster_env.sh`: downloads everything that
prepare_eval_data.py and train.py need into the cache directories, so they can run offline on the compute nodes.
"""
import argparse
import os

# Refuse to run without cluster_env.sh (checked before any library import reads these variables),
# otherwise everything would download to ~/.cache
for var in ["HF_HOME", "HF_HUB_CACHE", "DOCTR_CACHE_DIR", "TORCH_HOME", "FONT_DIR"]:
    assert os.environ.get(var), f"${var} is not set: run `source cluster_env.sh` first"
    print(f"{var} = {os.environ[var]}")

from diffusers.models import AutoencoderKL  # noqa: E402
from doctr.models import ocr_predictor  # noqa: E402
from torchmetrics.image.fid import FrechetInceptionDistance  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import text_data  # noqa: E402  (downloads the fonts into $FONT_DIR on import)

parser = argparse.ArgumentParser()
parser.add_argument("--ppl-model", type=str, default="Qwen/Qwen3-1.7B-Base")
args = parser.parse_args()

print(f"OpenWebText: {len(text_data.owt_files())} shards")  # ~24 GB of parquet into $HF_HUB_CACHE
for vae in ["mse", "ema"]:
    AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae}")
AutoTokenizer.from_pretrained(args.ppl_model)
AutoModelForCausalLM.from_pretrained(args.ppl_model)
ocr_predictor(pretrained=True)  # docTR weights -> $DOCTR_CACHE_DIR
FrechetInceptionDistance(feature=2048)  # Inception weights -> $TORCH_HOME
print("All assets downloaded.")
