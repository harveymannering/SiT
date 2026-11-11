"""
Run once on the login node (internet access) after `source cluster_env.sh`: downloads everything that
prepare_eval_data.py and train.py need, so they can run offline on the compute nodes.
"""
import argparse

from diffusers.models import AutoencoderKL
from doctr.models import ocr_predictor
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import AutoModelForCausalLM, AutoTokenizer

import text_data  # downloads the fonts into ./fonts on import

parser = argparse.ArgumentParser()
parser.add_argument("--ppl-model", type=str, default="Qwen/Qwen3-1.7B-Base")
args = parser.parse_args()

print(f"OpenWebText: {len(text_data.owt_files())} shards")  # ~24 GB of parquet into $HF_HOME
for vae in ["mse", "ema"]:
    AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae}")
AutoTokenizer.from_pretrained(args.ppl_model)
AutoModelForCausalLM.from_pretrained(args.ppl_model)
ocr_predictor(pretrained=True)  # docTR weights -> $DOCTR_CACHE_DIR
FrechetInceptionDistance(feature=2048)  # Inception weights -> $TORCH_HOME
print("All assets downloaded.")
