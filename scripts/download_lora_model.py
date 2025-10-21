#!/usr/bin/env python3
"""
Download LLaMAntino base model from Hugging Face and save locally.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

# Configuration
HF_MODEL_ID = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
LOCAL_MODEL_DIR = Path(__file__).parent.parent / "models" / "LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"


def download_base_model():
    """Download base model from Hugging Face to local directory."""
    print(f"Downloading base model from: {HF_MODEL_ID}")
    print(f"Saving to: {LOCAL_MODEL_DIR}")

    # Create local directory if it doesn't exist
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download model files
    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    print(f"✅ Base model downloaded successfully to: {LOCAL_MODEL_DIR}")
    print(f"\nYou can now load it with:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{LOCAL_MODEL_DIR}')")


if __name__ == "__main__":
    download_base_model()
