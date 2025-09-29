#!/usr/bin/env python3
"""
Model Merger CLI Tool

Merges PEFT/LoRA adapters with base models for optimized inference.
Usage: python merge_models.py --base minerva --peft marco_v3
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base model mappings - only models you actually use
BASE_MODELS = {
    "minerva": "sapienzanlp/Minerva-7B-base-v1.0",
}

# PEFT model mappings - your LoRA adapters
PEFT_MODELS = {
    "marco_v3": "marco/v3",
    "marco_v2": "marco/v2",
    "marco_v1": "marco/v1",
    "marco_v4_plus": "marco/v4_plus",
}


def get_project_root():
    """Get the project root directory (2 levels up from this script)"""
    return Path(__file__).parent.parent.parent


def resolve_model_path(model_name: str, is_base: bool = False) -> str:
    """
    Resolve model name to full path or HuggingFace model ID

    Args:
        model_name: Model name or path
        is_base: Whether this is a base model (use BASE_MODELS mapping)

    Returns:
        Full path or HuggingFace model ID
    """
    project_root = get_project_root()
    models_dir = project_root / "models"

    # Check if it's a base model alias
    if is_base and model_name.lower() in BASE_MODELS:
        logger.info(f"Using base model mapping: {model_name} -> {BASE_MODELS[model_name.lower()]}")
        return BASE_MODELS[model_name.lower()]

    # Check if it's a PEFT model alias
    if not is_base and model_name.lower() in PEFT_MODELS:
        logger.info(f"Using PEFT model mapping: {model_name} -> {PEFT_MODELS[model_name.lower()]}")
        model_name = PEFT_MODELS[model_name.lower()]

    # Check if it's a local path in models directory
    local_path = models_dir / model_name
    if local_path.exists():
        logger.info(f"Found local model: {local_path}")
        return str(local_path)

    # Check if it's an absolute path
    if os.path.isabs(model_name) and os.path.exists(model_name):
        logger.info(f"Using absolute path: {model_name}")
        return model_name

    # Assume it's a HuggingFace model ID
    logger.info(f"Treating as HuggingFace model ID: {model_name}")
    return model_name


def load_base_model(
    model_path: str,
    device_map: str = "auto",
    torch_dtype=torch.float16,
    use_quantization: bool = False,
):
    """
    Load the base model

    Args:
        model_path: Path or HuggingFace ID of base model
        device_map: Device mapping strategy
        torch_dtype: Torch data type
        use_quantization: Whether to use 4-bit quantization

    Returns:
        Loaded base model and tokenizer
    """
    logger.info(f"Loading base model: {model_path}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with offload support for large models
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            offload_folder="/tmp/offload",  # For memory-constrained environments
            low_cpu_mem_usage=True,
        )

        logger.info(f"✅ Base model loaded successfully")
        logger.info(f"   Model type: {type(model).__name__}")
        logger.info(
            f"   Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}"
        )

        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        raise


def load_and_merge_peft(base_model, peft_path: str):
    """
    Load PEFT adapter and merge with base model

    Args:
        base_model: Loaded base model
        peft_path: Path to PEFT adapter

    Returns:
        Merged model
    """
    logger.info(f"Loading PEFT adapter: {peft_path}")

    try:
        # Load PEFT model
        peft_model = PeftModel.from_pretrained(base_model, peft_path)
        logger.info(f"✅ PEFT adapter loaded successfully")

        # Merge the adapter weights
        logger.info("🔄 Merging PEFT adapter with base model...")
        merged_model = peft_model.merge_and_unload()
        logger.info(f"✅ Model merger completed successfully")

        return merged_model

    except Exception as e:
        logger.error(f"Failed to load/merge PEFT adapter: {e}")
        raise


def save_merged_model(model, tokenizer, output_path: str):
    """
    Save the merged model and tokenizer

    Args:
        model: Merged model
        tokenizer: Tokenizer
        output_path: Output directory path
    """
    logger.info(f"Saving merged model to: {output_path}")

    try:
        os.makedirs(output_path, exist_ok=True)

        # Save model
        model.save_pretrained(output_path)
        logger.info(f"✅ Model saved to {output_path}")

        # Save tokenizer
        tokenizer.save_pretrained(output_path)
        logger.info(f"✅ Tokenizer saved to {output_path}")

        # Save metadata
        metadata_path = os.path.join(output_path, "merge_info.txt")
        with open(metadata_path, "w") as f:
            f.write(f"Merged model information\n")
            f.write(f"Created: {torch.utils.data.get_worker_info()}\n")
            f.write(f"Model type: {type(model).__name__}\n")
            f.write(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

        logger.info(f"✅ Metadata saved to {metadata_path}")

    except Exception as e:
        logger.error(f"Failed to save merged model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Merge PEFT/LoRA adapters with base models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge Marco v3 LoRA with Minerva base model
  python merge_models.py --base minerva --peft marco_v3_lora_complete

  # Use custom output name
  python merge_models.py --base minerva --peft marco_v3_lora_complete --output marco_v3_merged

  # Use HuggingFace model IDs directly
  python merge_models.py --base Qwen/Qwen2.5-7B-Instruct --peft ./my_lora_adapter

Available base model aliases:
  minerva     -> sapienzanlp/Minerva-7B-base-v1.0

Available PEFT model aliases:
  marco_v3     -> marco/v3
  marco_v2     -> marco/v2
  marco_v1     -> marco/v1
  marco_v4_plus -> marco/v4_plus
        """,
    )

    parser.add_argument(
        "--base", help='Base model name/path (use aliases like "minerva" or full HF model IDs)'
    )

    parser.add_argument(
        "--peft", help="PEFT adapter name/path (looks in models/ directory or use full path)"
    )

    parser.add_argument("--output", help="Output directory name (default: {base}_{peft}_merged)")

    parser.add_argument(
        "--device-map", default="auto", help="Device mapping strategy (default: auto)"
    )

    parser.add_argument(
        "--torch-dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Torch data type (default: float16)",
    )

    parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Use 4-bit quantization to reduce memory usage (recommended for Colab)",
    )

    parser.add_argument(
        "--list-models", action="store_true", help="List available models in models/ directory"
    )

    args = parser.parse_args()

    # Handle list models
    if args.list_models:
        project_root = get_project_root()
        models_dir = project_root / "models"

        print("📁 Available models in models/ directory:")
        if models_dir.exists():
            for item in sorted(models_dir.iterdir()):
                if item.is_dir():
                    print(f"   📦 {item.name}")
        else:
            print("   (models/ directory not found)")

        print("\n🔗 Available base model aliases:")
        for alias, model_id in BASE_MODELS.items():
            print(f"   {alias:<12} -> {model_id}")

        print("\n📦 Available PEFT model aliases:")
        for alias, model_path in PEFT_MODELS.items():
            print(f"   {alias:<12} -> {model_path}")

        return

    # Check required arguments for merge operation
    if not args.base or not args.peft:
        parser.error("--base and --peft are required for model merging")

    try:
        # Resolve model paths
        base_model_path = resolve_model_path(args.base, is_base=True)
        peft_model_path = resolve_model_path(args.peft, is_base=False)

        # Set torch dtype
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map[args.torch_dtype]

        # Generate output path
        if args.output:
            output_name = args.output
        else:
            base_name = Path(args.base).name
            peft_name = Path(args.peft).name
            output_name = f"{base_name}_{peft_name}_merged"

        project_root = get_project_root()
        output_path = project_root / "models" / output_name

        logger.info("🚀 Starting model merge process...")
        logger.info(f"   Base model: {base_model_path}")
        logger.info(f"   PEFT adapter: {peft_model_path}")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Device map: {args.device_map}")
        logger.info(f"   Data type: {args.torch_dtype}")

        # Load base model
        base_model, tokenizer = load_base_model(
            base_model_path, device_map=args.device_map, torch_dtype=torch_dtype
        )

        # Load and merge PEFT adapter
        merged_model = load_and_merge_peft(base_model, peft_model_path)

        # Save merged model
        save_merged_model(merged_model, tokenizer, str(output_path))

        logger.info("🎉 Model merge completed successfully!")
        logger.info(f"📁 Merged model saved to: {output_path}")
        logger.info(f"💡 You can now use this model with vLLM or other inference engines")

    except Exception as e:
        logger.error(f"❌ Model merge failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
