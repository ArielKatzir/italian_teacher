#!/usr/bin/env python3
"""
Training script for Italian Exercise Generator using LLaMAntino-3-ANITA-8B

Usage:
  python train_exercise_generator.py                    # Use default config
  python train_exercise_generator.py --local-data       # Use local data paths
"""

import argparse
import logging
from pathlib import Path

from config_exercise_generation import (
    get_exercise_generation_config,
)
from lora_trainer import MarcoLoRATrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Italian Exercise Generator")
    parser.add_argument(
        "--local-data",
        action="store_true",
        help="Use local data paths instead of Colab paths",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/italian_exercise_generator_lora",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    args = parser.parse_args()

    # Load config
    config = get_exercise_generation_config()

    # Override with local paths if requested
    if args.local_data:
        base_path = Path(__file__).parent.parent.parent / "data" / "datasets" / "v4"
        config.data.train_file = str(base_path / "train.jsonl")
        config.data.validation_file = str(base_path / "validation.jsonl")
        config.data.test_file = str(base_path / "test.jsonl")
        logger.info(f"Using local data from: {base_path}")

    # Override other settings
    if args.no_wandb:
        config.experiment.use_wandb = False
        logger.info("Disabled Weights & Biases logging")

    config.training.output_dir = args.output_dir
    config.training.num_train_epochs = args.epochs

    # Display config summary
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model: {config.training.model_name}")
    logger.info(f"Train data: {config.data.train_file}")
    logger.info(f"Val data: {config.data.validation_file}")
    logger.info(f"Output dir: {config.training.output_dir}")
    logger.info(f"Epochs: {config.training.num_train_epochs}")
    logger.info(f"LoRA rank: {config.lora.r}, alpha: {config.lora.lora_alpha}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Batch size: {config.training.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"Max sequence length: {config.data.max_length}")
    logger.info("=" * 80)

    # Initialize and train
    try:
        logger.info("🚀 Starting training...")
        trainer = MarcoLoRATrainer(config=config)
        trainer.train()

        logger.info("=" * 80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"📁 Model saved to: {config.training.output_dir}")
        logger.info(f"🎯 You can now use this model for Italian exercise generation!")

    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ TRAINING FAILED!")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
