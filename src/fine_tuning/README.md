# Fine-tuning Infrastructure

This directory contains the infrastructure for fine-tuning specialized Italian teaching models using LoRA (Low-Rank Adaptation).

## Current Status: Phase 2.2 - LoRA Training Infrastructure

### Implemented Components ✅

1. **config.py** - Training configuration and hyperparameters ✅
2. **data_preprocessing.py** - Data loading and tokenization pipeline ✅
3. **lora_trainer.py** - Main training script with LoRA configuration ✅
4. **inference.py** - Inference utilities for the fine-tuned model ✅
5. **requirements.txt** - Dependencies for LoRA training ✅

### Planned Components

6. **evaluation.py** - Model evaluation and validation metrics
7. **question_generation.py** - Specialized question generation training

### Training Data

- **Source**: `/data/processed_llm_improved/` (10,130 samples with enhanced grammar explanations)
- **Base Model**: Qwen2.5-7B-Instruct
- **Training Method**: LoRA fine-tuning for parameter efficiency
- **Target**: Specialized Italian teaching conversation agent (Marco)

## Usage

### Training on Colab Pro

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run training:**
   ```python
   from src.fine_tuning.lora_trainer import MarcoLoRATrainer

   trainer = MarcoLoRATrainer()
   trainer.train()
   ```

3. **Use trained model:**
   ```python
   from src.fine_tuning.inference import MarcoInference

   marco = MarcoInference(lora_adapter_path="./marco_lora_checkpoints")
   response = marco.chat("Explain the grammar in 'Ho mangiato pizza'")
   ```

### Configuration

The training configuration is optimized for:
- **T4 GPU**: 1 batch size, 8 gradient accumulation steps, memory-efficient settings
- **A100 GPU**: 2 batch size, 4 gradient accumulation steps, higher throughput
- **LoRA**: rank=16, alpha=32, targeting 7 key modules for conversation tasks

### Next Steps

1. ✅ Set up Colab training environment
2. ✅ Configure PEFT library for LoRA
3. ✅ Implement training pipeline
4. 🔄 Run actual training on improved dataset
5. 📊 Add model evaluation metrics
6. 🎯 Add question generation capabilities