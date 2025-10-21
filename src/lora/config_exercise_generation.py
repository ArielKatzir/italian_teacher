"""
LoRA Training Configuration for Italian Exercise Generation Model

Optimized configuration for fine-tuning LLaMAntino-3-ANITA-8B
for Italian language exercise generation tasks.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LoRAConfig:
    """LoRA-specific configuration parameters."""

    # LoRA Parameters (V4: weaker alpha to preserve base model knowledge)
    r: int = 12  # LoRA rank - sweet spot between capacity (16) and regularization (8)
    lora_alpha: int = 6  # Weaker alpha (0.5x rank) to prevent catastrophic forgetting
    lora_dropout: float = 0.15  # Increased dropout - helps preserve base model knowledge

    # Target Modules (LLaMA 3 architecture)
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            # Target fewer modules to preserve more base knowledge (V4 optimization)
            self.target_modules = [
                "q_proj",  # Query projection (attention steering)
                "v_proj",  # Value projection (content transformation)
                # Removed k_proj, o_proj, gate_proj, up_proj, down_proj
                # to preserve more of the base model's Italian language knowledge
            ]


@dataclass
class TrainingConfig:
    """Training hyperparameters for exercise generation."""

    # Model Configuration - LLaMAntino-3-ANITA-8B (Italian-specialized, instruction-tuned)
    model_name: str = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    torch_dtype: str = "float16"  # Use float16 for memory efficiency
    attn_implementation: str = "flash_attention_2"  # Flash attention if available

    # Training Parameters (optimized for 3,983 examples)
    num_train_epochs: int = 1  # Single epoch to prevent overfitting and memorization
    per_device_train_batch_size: int = 8  # Adjust based on GPU
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 16

    # Learning Rate & Optimization (V3: balanced between V1 and V2)
    learning_rate: float = 1.5e-4  # Balanced LR between V1 (2e-4) and V2 (1e-4)
    weight_decay: float = 0.02  # Moderate L2 regularization (was 0.01 → 0.05)
    warmup_ratio: float = 0.1  # Standard warmup (back to V1 setting)
    lr_scheduler_type: str = "cosine"  # Smooth decay

    # Memory Optimization
    gradient_checkpointing: bool = True  # Essential for large models
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False  # Keep all columns for chat format

    # GPU Efficiency Optimizations
    tf32: bool = True  # Enable TensorFloat-32 for faster training on Ampere GPUs
    dataloader_num_workers: int = 4  # Parallel data loading
    ddp_find_unused_parameters: bool = False  # Optimize DDP performance

    # Logging & Saving (adjusted for 3,983 examples dataset)
    logging_steps: int = 50
    save_steps: int = 100  # ~8 checkpoints per epoch (reasonable for larger dataset)
    eval_steps: int = 100
    save_total_limit: int = 2  # Keep 3 best checkpoints

    # Output Configuration
    output_dir: str = "./models/italian_exercise_generator_v4"
    run_name: str = "italian_exercise_generator_v4"

    # Data Configuration
    max_seq_length: int = 2048  # Further increased to handle complex B2/C2 exercises (was 1536)
    data_seed: int = 42

    # Evaluation
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True


@dataclass
class DataConfig:
    """Data preprocessing configuration."""

    # Data Paths - V4 Augmented dataset (8,859 examples, comprehensive coverage)
    train_file: str = (
        "/content/drive/MyDrive/Colab Notebooks/italian_teacher/data/datasets/v4_augmented/train.jsonl"
    )
    validation_file: str = (
        "/content/drive/MyDrive/Colab Notebooks/italian_teacher/data/datasets/v4_augmented/validation.jsonl"
    )
    test_file: str = (
        "/content/drive/MyDrive/Colab Notebooks/italian_teacher/data/datasets/v4_augmented/test.jsonl"
    )

    # Preprocessing
    max_length: int = 2048  # V3: Further increased to handle complex B2/C2 exercises
    truncation: bool = True
    padding: str = "max_length"

    # Chat Template Configuration
    chat_template_format: str = "auto"  # Auto-detect or use fallback format
    add_generation_prompt: bool = True  # For inference

    # Data Filtering
    min_conversation_length: int = 2  # Minimum turns in conversation
    max_conversation_length: int = 10  # Maximum turns to prevent very long sequences


@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration."""

    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "italian-exercise-generator"
    wandb_entity: Optional[str] = None  # Your wandb username
    wandb_tags: List[str] = None

    # Experiment Info
    experiment_name: str = "llamantino_anita_exercise_gen_v4_preserve_base"
    description: str = (
        "V4 Weak LoRA (rank=12, alpha=6, 2 modules) - preserve base model Italian knowledge, prevent catastrophic forgetting"
    )

    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = [
                "italian",
                "exercise-generation",
                "lora",
                "llamantino-anita",
                "education",
                "v4-weak-alpha",
                "rank12-alpha6",
                "preserve-base-knowledge",
                "anti-catastrophic-forgetting",
            ]


# Complete Configuration
@dataclass
class FullConfig:
    """Complete configuration for LoRA training."""

    lora: LoRAConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    experiment: ExperimentConfig = None

    def __post_init__(self):
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()


def get_exercise_generation_config() -> FullConfig:
    """Get the default configuration for Italian exercise generation training."""
    return FullConfig()


def get_default_config() -> FullConfig:
    """Get the default configuration for Marco Italian teacher training."""
    return FullConfig()


# For GPU-specific adjustments
def adjust_config_for_gpu(config: FullConfig, gpu_name: str) -> FullConfig:
    """Adjust configuration based on available GPU."""

    if "T4" in gpu_name:
        # T4 15GB adjustments - Conservative settings
        config.training.per_device_train_batch_size = 2
        config.training.per_device_eval_batch_size = 2
        config.training.gradient_accumulation_steps = 8
        config.training.gradient_checkpointing = True
        config.training.dataloader_pin_memory = False  # Save GPU memory
        config.data.max_length = 1024
        print("🔧 T4 GPU detected: Using memory-optimized settings")

    elif "L4" in gpu_name:
        # L4 24GB adjustments - Memory-safe for 8B model with 4-bit quantization
        config.training.per_device_train_batch_size = 1
        config.training.per_device_eval_batch_size = 1
        config.training.gradient_accumulation_steps = 16  # Effective batch size: 16
        config.training.gradient_checkpointing = True
        config.training.dataloader_pin_memory = True
        config.data.max_length = 1024
        config.training.eval_steps = 200
        config.training.save_steps = 200
        config.training.logging_steps = 50
        config.training.dataloader_num_workers = 2
        config.training.tf32 = True
        print("🚀 L4 GPU detected: Using memory-safe settings for 8B model")

    elif "A100" in gpu_name:
        # A100 40GB/80GB adjustments - Optimized for 8B model with LoRA
        config.training.per_device_train_batch_size = 4
        config.training.per_device_eval_batch_size = 4
        config.training.gradient_accumulation_steps = 4  # Effective batch size: 16
        config.training.gradient_checkpointing = True
        config.training.dataloader_pin_memory = True
        config.data.max_length = 2048  # V3: Increased to prevent B2/C2 truncation
        config.training.dataloader_num_workers = 8
        config.training.tf32 = True  # Enable TF32 for faster training
        print("🏎️  A100 GPU detected: Using high-performance settings for 8B model")

    else:
        # Unknown GPU - Use conservative T4-like settings
        config.training.per_device_train_batch_size = 2
        config.training.per_device_eval_batch_size = 2
        config.training.gradient_accumulation_steps = 8
        config.training.gradient_checkpointing = True
        config.training.dataloader_pin_memory = False
        config.data.max_length = 1024
        print(f"❓ Unknown GPU ({gpu_name}): Using conservative settings")

    # Calculate and display effective batch size
    effective_batch_size = (
        config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
    )
    print(f"   Effective batch size: {effective_batch_size}")
    print(
        f"   Memory optimization: {'Enabled' if config.training.gradient_checkpointing else 'Disabled'}"
    )

    return config
