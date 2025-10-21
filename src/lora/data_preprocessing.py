"""
Data preprocessing pipeline for Italian teaching conversations.

Handles loading, tokenization, and formatting of conversational data
for LoRA fine-tuning of Qwen2.5-7B-Instruct.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

try:
    # Try relative imports first (when used as package)
    from .config import DataConfig, FullConfig
except ImportError:
    # Fall back to direct imports (when used standalone)
    from config import DataConfig, FullConfig

logger = logging.getLogger(__name__)


class ConversationDataProcessor:
    """Processes conversation data for Italian teaching model training."""

    def __init__(
        self, config: DataConfig, tokenizer_name: str = "sapienzanlp/Minerva-7B-base-v1.0"
    ):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        # Set up tokenizer for chat format
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Initialized tokenizer: {tokenizer_name}")
        logger.info(f"Vocab size: {len(self.tokenizer)}")
        logger.info(f"Max length: {config.max_length}")

    def load_jsonl_data(self, file_path: str) -> List[Dict]:
        """Load conversation data from JSONL file."""
        data = []
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return data

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        if self.validate_conversation(item):
                            data.append(item)
                        else:
                            logger.debug(f"Skipped invalid conversation at line {line_num}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue

            logger.info(f"Loaded {len(data)} valid conversations from {file_path}")
            return data

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

    def validate_conversation(self, item: Dict) -> bool:
        """Validate conversation structure and content."""
        # Check required fields
        if "messages" not in item:
            return False

        messages = item["messages"]
        if not isinstance(messages, list):
            return False

        # Check conversation length
        if len(messages) < self.config.min_conversation_length:
            return False
        if len(messages) > self.config.max_conversation_length:
            return False

        # Validate message structure
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if msg["role"] not in ["user", "assistant", "system"]:
                return False
            if not isinstance(msg["content"], str) or len(msg["content"].strip()) == 0:
                return False

        return True

    def format_conversation_for_training(self, conversation: List[Dict]) -> str:
        """Format conversation for training with fallback for models without chat templates."""

        # Try to use model's chat template if available
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            try:
                formatted = self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
                return formatted
            except Exception as e:
                print(f"Chat template failed: {e}, falling back to simple format")

        # Fallback: Simple format for models without chat templates (like Minerva base)
        formatted_parts = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                formatted_parts.append(f"### User:\n{content}\n")
            elif role == "assistant":
                formatted_parts.append(f"### Assistant:\n{content}\n")
            elif role == "system":
                formatted_parts.append(f"### System:\n{content}\n")

        return "".join(formatted_parts)

    def tokenize_conversation(self, formatted_text: str) -> Dict[str, Any]:
        """Tokenize formatted conversation with proper attention mask."""

        # Tokenize the conversation
        tokenized = self.tokenizer(
            formatted_text,
            truncation=True,
            padding=False,  # We'll pad in collate_fn
            max_length=self.config.max_length,
            return_tensors=None,  # Return lists, not tensors
        )

        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def process_dataset_item(self, item: Dict) -> Optional[Dict[str, Any]]:
        """Process a single dataset item."""
        try:
            messages = item["messages"]

            # Format conversation
            formatted_text = self.format_conversation_for_training(messages)

            # Tokenize
            tokenized = self.tokenize_conversation(formatted_text)

            # Add metadata
            result = {
                **tokenized,
                "conversation_id": item.get("metadata", {}).get("conversation_id", "unknown"),
                "source": item.get("metadata", {}).get("source", "unknown"),
                "level": item.get("metadata", {}).get("level", "unknown"),
                "topic": item.get("metadata", {}).get("topic", "unknown"),
                "original_length": len(messages),
            }

            return result

        except Exception as e:
            logger.warning(f"Error processing conversation: {e}")
            return None

    def create_dataset(self, data: List[Dict]) -> Dataset:
        """Create HuggingFace Dataset from conversation data."""

        processed_items = []

        for item in data:
            processed = self.process_dataset_item(item)
            if processed is not None:
                processed_items.append(processed)

        if not processed_items:
            raise ValueError("No valid conversations found in dataset")

        # Create dataset
        dataset = Dataset.from_list(processed_items)

        # Log statistics
        logger.info(f"Created dataset with {len(dataset)} examples")
        logger.info(
            f"Average input length: {sum(len(ex['input_ids']) for ex in dataset) / len(dataset):.1f}"
        )

        # Log level distribution
        level_counts = {}
        for ex in dataset:
            level = ex["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
        logger.info(f"CEFR level distribution: {level_counts}")

        return dataset

    def load_all_datasets(self) -> DatasetDict:
        """Load and process all datasets (train, validation, test)."""

        datasets = {}

        # Load training data
        if Path(self.config.train_file).exists():
            train_data = self.load_jsonl_data(self.config.train_file)
            if train_data:
                datasets["train"] = self.create_dataset(train_data)
                logger.info(f"Training dataset: {len(datasets['train'])} examples")

        # Load validation data
        if Path(self.config.validation_file).exists():
            val_data = self.load_jsonl_data(self.config.validation_file)
            if val_data:
                datasets["validation"] = self.create_dataset(val_data)
                logger.info(f"Validation dataset: {len(datasets['validation'])} examples")

        # Load test data
        if Path(self.config.test_file).exists():
            test_data = self.load_jsonl_data(self.config.test_file)
            if test_data:
                datasets["test"] = self.create_dataset(test_data)
                logger.info(f"Test dataset: {len(datasets['test'])} examples")

        if not datasets:
            raise ValueError("No datasets loaded successfully")

        return DatasetDict(datasets)


class DataCollator:
    """Custom data collator for conversation data."""

    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""

        # Extract sequences
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad sequences
        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids}, padding=True, max_length=self.max_length, return_tensors="pt"
        )

        labels = self.tokenizer.pad(
            {"input_ids": labels}, padding=True, max_length=self.max_length, return_tensors="pt"
        )["input_ids"]

        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids["input_ids"],
            "attention_mask": input_ids["attention_mask"],
            "labels": labels,
        }


def prepare_datasets(config: FullConfig) -> tuple[DatasetDict, DataCollator]:
    """Prepare datasets and data collator for training."""

    # Initialize processor
    processor = ConversationDataProcessor(config.data, config.training.model_name)

    # Load datasets
    datasets = processor.load_all_datasets()

    # Create data collator
    collator = DataCollator(processor.tokenizer, config.data.max_length)

    logger.info("Dataset preparation complete")
    logger.info(f"Available splits: {list(datasets.keys())}")

    return datasets, collator


# Example usage for testing
if __name__ == "__main__":
    try:
        from .config import get_default_config
    except ImportError:
        from config import get_default_config

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Test with default config
    config = get_default_config()

    try:
        datasets, collator = prepare_datasets(config)
        print("✅ Data preprocessing pipeline successful!")
        print(f"Datasets: {list(datasets.keys())}")
        for split, dataset in datasets.items():
            print(f"{split}: {len(dataset)} examples")

    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        raise
