"""
LoRA Training Script for Marco Italian Teaching Model

Fine-tunes Qwen2.5-7B-Instruct using LoRA for Italian conversation and teaching.
Designed to run on Colab Pro with T4/A100 GPUs.
"""

import logging
from pathlib import Path
from typing import Optional

import torch

# LoRA libraries
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# Quantization for memory efficiency
# Core training libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Experiment tracking
import wandb

# Local imports
try:
    # Try relative imports first (when used as package)
    from .config import FullConfig, adjust_config_for_gpu, get_default_config
    from .data_preprocessing import prepare_datasets
except ImportError:
    # Fall back to direct imports (when used standalone)
    from config import FullConfig, adjust_config_for_gpu, get_default_config
    from data_preprocessing import prepare_datasets

logger = logging.getLogger(__name__)


class MarcoLoRATrainer:
    """LoRA trainer for Marco Italian teaching model."""

    def __init__(self, config: Optional[FullConfig] = None):
        """Initialize trainer with configuration."""

        self.config = config or get_default_config()

        # Auto-adjust for available GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Detected GPU: {gpu_name}")
            self.config = adjust_config_for_gpu(self.config, gpu_name)
        else:
            logger.warning("No GPU detected. Training will be very slow.")

        # Set random seed for reproducibility
        set_seed(self.config.training.data_seed)

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.datasets = None
        self.data_collator = None
        self.trainer = None

    def setup_model_and_tokenizer(self):
        """Load and configure model and tokenizer."""

        logger.info(f"Loading model: {self.config.training.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.training.model_name, trust_remote_code=True
        )

        # Set up padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.training.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        logger.info("Model and tokenizer loaded successfully")

    def setup_lora(self):
        """Configure and apply LoRA to the model."""

        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    def setup_data(self):
        """Prepare datasets and data collator."""

        logger.info("Preparing datasets...")
        self.datasets, self.data_collator = prepare_datasets(self.config)

        logger.info("Datasets prepared successfully")

    def setup_training_arguments(self) -> TrainingArguments:
        """Create training arguments."""

        return TrainingArguments(
            output_dir=self.config.training.output_dir,
            # Training schedule
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            # Optimization
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            # Memory optimization
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            remove_unused_columns=self.config.training.remove_unused_columns,
            # Logging and evaluation
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            eval_strategy=self.config.training.eval_strategy,
            # Saving
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            # Experiment tracking
            run_name=self.config.training.run_name,
            report_to="wandb" if self.config.experiment.use_wandb else None,
            # Hardware optimization
            fp16=True,  # Use mixed precision
            dataloader_num_workers=2,
            # Reproducibility
            seed=self.config.training.data_seed,
        )

    def setup_wandb(self):
        """Initialize Weights & Biases tracking."""

        if not self.config.experiment.use_wandb:
            return

        wandb.init(
            project=self.config.experiment.wandb_project,
            entity=self.config.experiment.wandb_entity,
            name=self.config.experiment.experiment_name,
            tags=self.config.experiment.wandb_tags,
            config={
                "model": self.config.training.model_name,
                "lora_r": self.config.lora.r,
                "lora_alpha": self.config.lora.lora_alpha,
                "learning_rate": self.config.training.learning_rate,
                "batch_size": self.config.training.per_device_train_batch_size,
                "gradient_accumulation_steps": self.config.training.gradient_accumulation_steps,
                "epochs": self.config.training.num_train_epochs,
                "max_length": self.config.data.max_length,
            },
        )

        logger.info("Weights & Biases initialized")

    def train(self):
        """Run the complete training pipeline."""

        logger.info("Starting Marco LoRA training pipeline...")

        # Setup components
        self.setup_wandb()
        self.setup_model_and_tokenizer()
        self.setup_lora()
        self.setup_data()

        # Create training arguments
        training_args = self.setup_training_arguments()

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets.get("validation"),
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )

        # Start training
        logger.info("🚀 Starting training...")

        try:
            train_result = self.trainer.train(resume_from_checkpoint=True)

            # Save final model
            logger.info("💾 Saving final model...")
            self.trainer.save_model()

            # Log training summary
            logger.info("✅ Training completed successfully!")
            logger.info(f"Final train loss: {train_result.training_loss:.4f}")

            if self.config.experiment.use_wandb:
                wandb.finish()

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            if self.config.experiment.use_wandb:
                wandb.finish(exit_code=1)
            raise

    def save_model(self, output_path: str):
        """Save the trained LoRA model."""

        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"Model saved to: {output_path}")


def main():
    """Main training function for use in notebooks or scripts."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize and run training
        trainer = MarcoLoRATrainer()
        trainer.train()

        print("🎉 Marco LoRA training completed successfully!")
        print(f"📁 Model saved to: {trainer.config.training.output_dir}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
