# Fine-tuning Infrastructure

This directory contains the infrastructure for fine-tuning specialized Italian teaching models using LoRA (Low-Rank Adaptation).

## Current Status: Phase 2.2 - LoRA Training Infrastructure

### Planned Components

1. **config.py** - Training configuration and hyperparameters
2. **data_preprocessing.py** - Data loading and tokenization pipeline
3. **lora_trainer.py** - Main training script with LoRA configuration
4. **evaluation.py** - Model evaluation and validation metrics
5. **inference.py** - Inference utilities for the fine-tuned model
6. **question_generation.py** - Specialized question generation training

### Training Data

- **Source**: `/data/processed_llm_improved/` (10,130 samples with enhanced grammar explanations)
- **Base Model**: Qwen2.5-7B-Instruct
- **Training Method**: LoRA fine-tuning for parameter efficiency
- **Target**: Specialized Italian teaching conversation agent (Marco)

### Next Steps

1. Set up Colab training environment
2. Configure PEFT library for LoRA
3. Implement training pipeline
4. Add question generation capabilities