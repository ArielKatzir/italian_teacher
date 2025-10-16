# Fine-tuning Infrastructure

This directory contains the infrastructure for fine-tuning specialized Italian teaching models using LoRA (Low-Rank Adaptation).

## Current Status: Phase 2.5 - Marco v2 Dual-LLM Training

### 🔄 **Dual-LLM Strategy: Qwen → Minerva**

**Problem Solved**: Eliminated circular training and template overfitting from Marco v1

**Approach**:
1. **Data Generation**: Qwen/Qwen2.5-3B-Instruct generates diverse Italian teaching responses
2. **Model Training**: Minerva-7B-base-v1.0 (Italian-specialized) learns from Qwen's responses
3. **Result**: Marco v2 with authentic Italian foundations + creative teaching capabilities

### Implemented Components ✅

1. **config.py** - Training configuration optimized for Minerva-7B ✅
2. **data_preprocessing.py** - Data loading and tokenization pipeline ✅
3. **lora_trainer.py** - Main training script with LoRA configuration ✅
4. **inference.py** - Inference utilities for the fine-tuned model ✅
5. **requirements.txt** - Dependencies for LoRA training ✅
6. **marco_lora_training.ipynb** - Complete Colab training notebook ✅

### Planned Components

6. **evaluation.py** - Model evaluation and validation metrics
7. **question_generation.py** - Specialized question generation training

### Training Data

- **Source**: `/data/processed/complete/` (~17K samples with Qwen-generated responses)
- **Base Model**: sapienzanlp/Minerva-7B-base-v1.0 (Italian-specialized)
- **Training Method**: LoRA fine-tuning for parameter efficiency
- **Target**: Marco v2 - Italian teaching conversation agent

## Usage

### Training on Colab Pro

**Prerequisites**: Complete dataset with Qwen-generated responses

1. **Prepare data**: Run `data/COLAB_generate_assistant_message_LLM.ipynb` to fill blank responses
2. **Run training**: Use `marco_lora_training.ipynb` for complete training pipeline

**Quick Setup**:
```python
from src.fine_tuning.lora_trainer import MarcoLoRATrainer
from src.fine_tuning.config import get_default_config

config = get_default_config()  # Uses Minerva-7B + complete dataset
trainer = MarcoLoRATrainer(config=config)
trainer.train()
```

**Use trained model:**
```python
from src.fine_tuning.inference import MarcoInference

marco = MarcoInference(lora_adapter_path="./models/marco_lora_minerva_v2")
response = marco.chat("Explain the grammar in 'Ho mangiato pizza'")
```

### Configuration

Optimized for **Dual-LLM Training**:
- **Base Model**: Minerva-7B-base-v1.0 (Italian-specialized)
- **Training Data**: ~17K conversations with Qwen-generated responses
- **T4 GPU**: 1 batch size, 8 gradient accumulation steps (8-10 hours)
- **L4 GPU**: 3 batch size, 3 gradient accumulation steps (4-5 hours) ⭐ **RECOMMENDED**
- **A100 GPU**: 2 batch size, 4 gradient accumulation steps (2-3 hours)
- **LoRA**: rank=16, alpha=32, targeting 7 key modules for Llama architecture

### Quality Improvements

**Marco v2 vs Marco v1**:
- ✅ **Zero template overfitting** (eliminated pattern matching)
- ✅ **Italian-specialized base** (Minerva vs general Qwen)
- ✅ **Diverse teaching styles** (Qwen creativity + Minerva accuracy)
- ✅ **Authentic language patterns** (no circular training)
- ✅ **17K complete conversations** (vs 10K incomplete)

### Next Steps

1. ✅ Complete dual-LLM data generation pipeline
2. ✅ Update training infrastructure for Minerva
3. ✅ Create optimized Colab training notebook
4. 🔄 **Ready for Marco v2 training!**
5. 📊 Add model evaluation metrics
6. 🎯 Add question generation capabilities