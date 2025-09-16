#!/usr/bin/env python3
"""
Model Download Script for Italian Teacher

Downloads and caches Llama models with Hugging Face authentication.
Run this once to download models, then use them offline.

Usage:
    python scripts/download_models.py --token YOUR_HF_TOKEN
    python scripts/download_models.py --model llama3.1-3b
    HF_TOKEN=your_token python scripts/download_models.py --all
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import torch
    from huggingface_hub import login, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("❌ Missing dependencies. Please install:")
    print("pip install huggingface_hub transformers torch")
    sys.exit(1)

# Model configurations - accessible open source models (no gates!)
MODELS = {
    "gemma-2b": {
        "repo_id": "google/gemma-2b-it",
        "description": "Google Gemma 2B - Very fast, good for development and testing",
        "size": "~5GB",
        "memory_req": "~1GB with 4-bit quantization",
        "license_url": "https://huggingface.co/google/gemma-2b-it",
        "gated": False,
    },
    "phi3-mini": {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Microsoft Phi-3 3.8B - Excellent quality, no access restrictions",
        "size": "~7GB",
        "memory_req": "~2GB with 4-bit quantization",
        "license_url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
        "gated": False,
    },
    "mistral-7b": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral 7B - Excellent multilingual, best quality",
        "size": "~14GB",
        "memory_req": "~4GB with 4-bit quantization",
        "license_url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
        "gated": False,
    },
}


def authenticate_hf(token=None):
    """Authenticate with Hugging Face."""
    if token:
        print(f"🔐 Using provided token: {token[:8]}...")
        login(token=token)
    elif os.getenv("HF_TOKEN"):
        print("🔐 Using HF_TOKEN environment variable")
        login(token=os.getenv("HF_TOKEN"))
    else:
        print("🔐 Attempting to use cached HF credentials...")
        try:
            # Try to use existing credentials
            from huggingface_hub import HfApi

            api = HfApi()
            user = api.whoami()
            print(f"✅ Authenticated as: {user['name']}")
        except Exception:
            print("❌ No valid HF authentication found.")
            print("Please provide a token via:")
            print("  --token YOUR_TOKEN")
            print("  export HF_TOKEN=YOUR_TOKEN")
            print("  huggingface-cli login")
            return False
    return True


def check_gpu_memory():
    """Check available GPU memory."""
    if not torch.cuda.is_available():
        print("💻 No GPU detected - models will use CPU")
        return 0

    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    return gpu_memory


def download_model(model_key, force=False):
    """Download and cache a specific model."""
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available models: {list(MODELS.keys())}")
        return False

    model_info = MODELS[model_key]
    repo_id = model_info["repo_id"]

    print(f"\n🔽 Downloading {model_key}:")
    print(f"  📦 Repository: {repo_id}")
    print(f"  📊 Size: {model_info['size']}")
    print(f"  💾 Memory: {model_info['memory_req']}")
    print(f"  📝 Description: {model_info['description']}")

    # Check if model requires special access (these models are all open!)
    if model_info.get("gated", False):
        print(f"\n⚠️  This model may require license acceptance!")
        print(f"🔗 Please visit: {model_info['license_url']}")
        print(f"📋 Click 'Agree and access repository' if prompted")
    else:
        print(f"\n✅ This model is fully open source - no special permissions needed!")

    try:
        # Create cache directory
        cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Cache directory: {cache_dir}")

        # Download model files
        print("⬇️  Downloading model files...")
        model_path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=False)

        # Download tokenizer
        print("⬇️  Downloading tokenizer...")
        AutoTokenizer.from_pretrained(repo_id)

        print(f"✅ {model_key} downloaded successfully!")
        print(f"📍 Cached at: {model_path}")

        # Test loading (optional)
        if input("🧪 Test load model? (y/N): ").lower().startswith("y"):
            print("🔄 Testing model load...")
            try:
                # Load with 4-bit quantization to save memory
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )

                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                print("✅ Model loads correctly!")

                # Clean up memory
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️  Model load test failed: {e}")
                print("💡 Model downloaded but may need different loading config")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Italian Teacher models")
    parser.add_argument("--token", help="Hugging Face token")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Specific model to download")
    parser.add_argument("--all", action="store_true", help="Download all available models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")

    args = parser.parse_args()

    if args.list:
        print("📋 Available models:")
        for key, info in MODELS.items():
            print(f"\n🤖 {key}:")
            print(f"  📦 {info['repo_id']}")
            print(f"  📊 Size: {info['size']}")
            print(f"  💾 Memory: {info['memory_req']}")
            print(f"  📝 {info['description']}")
        return

    print("🇮🇹 Italian Teacher - Model Download Script")
    print("=" * 50)

    # Check GPU
    check_gpu_memory()

    # Authenticate
    if not authenticate_hf(args.token):
        return

    # Download models
    if args.all:
        print(f"\n📦 Downloading all {len(MODELS)} models...")
        for model_key in MODELS:
            if not download_model(model_key, args.force):
                print(f"⚠️  Failed to download {model_key}, continuing...")
    elif args.model:
        download_model(args.model, args.force)
    else:
        # Interactive mode
        print("\n🤖 Available models:")
        for i, (key, info) in enumerate(MODELS.items(), 1):
            print(f"  {i}. {key} - {info['description']} ({info['size']})")

        try:
            choice = input("\n🔽 Enter model number to download (or 'all'): ").strip()

            if choice.lower() == "all":
                for model_key in MODELS:
                    download_model(model_key, args.force)
            else:
                model_keys = list(MODELS.keys())
                if choice.isdigit() and 1 <= int(choice) <= len(model_keys):
                    model_key = model_keys[int(choice) - 1]
                    download_model(model_key, args.force)
                else:
                    print("❌ Invalid choice")
        except KeyboardInterrupt:
            print("\n👋 Download cancelled")


if __name__ == "__main__":
    main()
