"""
LoRA Training Configuration for Italian Teaching Model (Marco)

Optimized configuration for fine-tuning Qwen2.5-7B-Instruct
for Italian conversation and teaching tasks.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LoRAConfig:
    """LoRA-specific configuration parameters."""

    # LoRA Parameters (optimized for conversation tasks)
    r: int = 16  # LoRA rank - balance between performance and efficiency
    lora_alpha: int = 32  # LoRA scaling parameter (2x rank is standard)
    lora_dropout: float = 0.1  # Dropout for LoRA layers

    # Target Modules (Qwen2.5 specific)
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            # Target attention and feed-forward layers for Qwen2.5
            self.target_modules = [
                "q_proj",  # Query projection
                "k_proj",  # Key projection
                "v_proj",  # Value projection
                "o_proj",  # Output projection
                "gate_proj",  # Gate projection (MLP)
                "up_proj",  # Up projection (MLP)
                "down_proj",  # Down projection (MLP)
            ]


@dataclass
class TrainingConfig:
    """Training hyperparameters for conversational AI."""

    # Model Configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "float16"  # Use float16 for memory efficiency
    attn_implementation: str = "flash_attention_2"  # Flash attention if available

    # Training Parameters
    num_train_epochs: int = 3  # Conservative for LoRA
    per_device_train_batch_size: int = 1  # Small batch for memory efficiency
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch size = 8

    # Learning Rate & Optimization
    learning_rate: float = 2e-4  # Higher LR for LoRA (vs 5e-5 for full fine-tuning)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  # 10% warmup steps
    lr_scheduler_type: str = "cosine"

    # Memory Optimization
    gradient_checkpointing: bool = True  # Essential for large models
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False  # Keep all columns for chat format

    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2  # Keep only 2 checkpoints

    # Output Configuration
    output_dir: str = "./models/marco_lora_v2"
    run_name: str = "marco_italian_teacher"

    # Data Configuration
    max_seq_length: int = 1024  # Max conversation length
    data_seed: int = 42

    # Evaluation
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True


@dataclass
class DataConfig:
    """Data preprocessing configuration."""

    # Data Paths
    train_file: str = (
        "/content/drive/MyDrive/Colab Notebooks/italian_teacher/data/processed_llm_improved/train.jsonl"
    )
    validation_file: str = (
        "/content/drive/MyDrive/Colab Notebooks/italian_teacher/data/processed_llm_improved/validation.jsonl"
    )
    test_file: str = (
        "/content/drive/MyDrive/Colab Notebooks/italian_teacher/data/processed_llm_improved/test.jsonl"
    )

    # Preprocessing
    max_length: int = 1024  # Maximum sequence length
    truncation: bool = True
    padding: str = "max_length"

    # Chat Template Configuration
    chat_template_format: str = "qwen"  # Use Qwen's chat format
    add_generation_prompt: bool = True  # For inference

    # Data Filtering
    min_conversation_length: int = 2  # Minimum turns in conversation
    max_conversation_length: int = 10  # Maximum turns to prevent very long sequences


@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration."""

    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "italian-teacher-marco"
    wandb_entity: Optional[str] = None  # Your wandb username
    wandb_tags: List[str] = None

    # Experiment Info
    experiment_name: str = "marco_qwen2.5_lora_v1"
    description: str = "LoRA fine-tuning of Qwen2.5-7B for Italian teaching conversations"

    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = [
                "italian",
                "teaching",
                "lora",
                "qwen2.5",
                "conversation",
                "education",
                "marco",
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


def get_default_config() -> FullConfig:
    """Get the default configuration for Marco Italian teacher training."""
    return FullConfig()


# For GPU-specific adjustments
def adjust_config_for_gpu(config: FullConfig, gpu_name: str) -> FullConfig:
    """Adjust configuration based on available GPU."""

    if "T4" in gpu_name:
        # T4 15GB adjustments - Conservative settings
        config.training.per_device_train_batch_size = 1
        config.training.per_device_eval_batch_size = 1
        config.training.gradient_accumulation_steps = 8
        config.training.gradient_checkpointing = True
        config.training.dataloader_pin_memory = False  # Save GPU memory
        config.data.max_length = 1024
        print("🔧 T4 GPU detected: Using memory-optimized settings")

    elif "L4" in gpu_name:
        # L4 24GB adjustments - Optimal performance balance
        config.training.per_device_train_batch_size = 3
        config.training.per_device_eval_batch_size = 4  # Can be higher for eval
        config.training.gradient_accumulation_steps = 3  # Effective batch size: 9
        config.training.gradient_checkpointing = True  # Still beneficial
        config.training.dataloader_pin_memory = True  # Plenty of memory
        config.data.max_length = 1024
        config.training.eval_steps = 100  # More frequent evaluation
        config.training.logging_steps = 5  # More detailed logging
        print("🚀 L4 GPU detected: Using high-performance settings")

    elif "A100" in gpu_name:
        # A100 40GB/80GB adjustments - Maximum performance
        config.training.per_device_train_batch_size = 2
        config.training.per_device_eval_batch_size = 3
        config.training.gradient_accumulation_steps = 4
        config.training.gradient_checkpointing = True
        config.training.dataloader_pin_memory = True
        config.data.max_length = 1024
        print("🏎️  A100 GPU detected: Using maximum performance settings")

    else:
        # Unknown GPU - Use conservative T4-like settings
        config.training.per_device_train_batch_size = 1
        config.training.per_device_eval_batch_size = 1
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
