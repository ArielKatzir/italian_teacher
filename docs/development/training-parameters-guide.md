# LoRA Training Parameters Deep Dive

Comprehensive guide to all training parameters used in Marco's Italian teaching model fine-tuning.

## Table of Contents
- [Dual-LLM Training Strategy](#dual-llm-training-strategy)
- [Training Schedule](#training-schedule)
- [Optimization Parameters](#optimization-parameters)
- [Memory Optimization](#memory-optimization)
- [Logging and Evaluation](#logging-and-evaluation)
- [Model Saving](#model-saving)
- [Experiment Tracking](#experiment-tracking)
- [Hardware Optimization](#hardware-optimization)
- [Reproducibility](#reproducibility)

---

## Dual-LLM Training Strategy

### Overview: Generate with Qwen, Train on Minerva

Starting with Marco v2, we implement a **dual-LLM approach** to avoid circular training problems and maximize training data quality.

### The Problem with Single-LLM Approach (Marco v1)

**What we did before**:
- Used the same model (Qwen) to both generate training responses AND as the base for fine-tuning
- Created synthetic training data where the model was essentially training on its own outputs
- Generated template-based responses with pattern matching artifacts

**Why it was problematic**:
1. **Circular Training**: Model learns to mimic its own generation patterns rather than authentic teaching styles
2. **Template Overfitting**: Responses became formulaic (e.g., "This A1-level text shows developing proficiency...")
3. **Limited Diversity**: Same model architecture led to repetitive response structures
4. **Echo Chamber Effect**: Model reinforced its own biases and limitations

### The Solution: Dual-LLM Architecture

**What we do now (Marco v2)**:

#### Phase 1: Response Generation with Qwen
- **Model**: Qwen/Qwen2.5-3B-Instruct
- **Purpose**: Generate diverse, level-appropriate Italian teaching responses
- **Input**: Authentic user questions from CELI, CIMA, and Tatoeba corpora
- **Output**: 17,142 teaching responses covering A1-C2 levels
- **Quality**: Level-specific prompts ensure appropriate complexity and teaching style

#### Phase 2: LoRA Training on Minerva
- **Model**: sapienzanlp/Minerva-3B-base-v1.0 (Italian-specialized)
- **Purpose**: Fine-tune Italian language model for teaching conversations
- **Input**: Qwen-generated responses paired with authentic user questions
- **Output**: Marco v2 - Italian teaching specialist

### Why This Approach Works

1. **Cross-Model Diversity**: Qwen's generation style differs from Minerva's base architecture
2. **Language Specialization**: Minerva has strong Italian foundations, Qwen provides teaching creativity
3. **Authentic Data Foundation**: All user questions come from real learner corpora (CELI exams, CIMA tutoring)
4. **No Template Artifacts**: Completely eliminated pattern-matching responses
5. **Level Expertise**: Each CEFR level gets appropriate teaching approach

### Implementation Details

**Generation Stage (Qwen)**:
```python
# Level-specific prompt engineering
level_approaches = {
    "A1": "friendly, patient Italian teacher helping an absolute beginner",
    "B2": "skilled Italian teacher helping an upper-intermediate student",
    "C2": "master Italian teacher helping a near-native speaker"
}

# Dynamic generation parameters by level
generation_params = {
    "A1": {"max_new_tokens": 150, "temperature": 0.6},  # Shorter, focused
    "C2": {"max_new_tokens": 300, "temperature": 0.8}   # Longer, sophisticated
}
```

**Training Stage (Minerva)**:
- Standard LoRA fine-tuning parameters (documented below)
- Italian-optimized base model provides strong linguistic foundation
- Cross-model training data prevents overfitting to generation patterns

### Quality Metrics

**Dataset Composition (17,913 total conversations)**:
- **771 authentic responses** (4.3%) - preserved from CIMA human tutors
- **17,142 generated responses** (95.7%) - created by Qwen with level-appropriate prompts
- **0 template responses** - completely eliminated pattern matching

**Level Distribution**:
- A1: 4,000 (22.3%) | A2: 4,000 (22.3%) | B1: 3,501 (19.5%)
- B2: 2,600 (14.5%) | C1: 2,600 (14.5%) | C2: 1,212 (6.8%)

This dual-LLM strategy ensures Marco v2 learns from diverse, high-quality teaching examples while maintaining authentic Italian language patterns.

---

## Training Schedule

### `num_train_epochs=3`
**What it does**: Number of complete passes through the training dataset.

**Why 3 epochs**:
- **Too few (1-2)**: Model might underfit, not learning enough patterns
- **Just right (3-5)**: LoRA typically converges quickly, 3 epochs balance learning vs overfitting
- **Too many (6+)**: Risk of overfitting to training data, especially with 10K samples

**Research evidence**: Most successful LoRA papers use 3-5 epochs for conversational tasks.

### `per_device_train_batch_size=1` (T4) / `2` (A100)
**What it does**: Number of training samples processed simultaneously on each GPU.

**GPU-Specific Values**:
```python
# T4 GPU (15GB memory)
per_device_train_batch_size = 1
# Reason: T4 has limited memory, batch_size=2 would cause OOM

# L4 GPU (24GB memory) - RECOMMENDED
per_device_train_batch_size = 3
# Reason: L4 has excellent memory/performance balance for LoRA

# A100 GPU (40GB memory)
per_device_train_batch_size = 2
# Reason: A100 optimized for maximum performance
```

**Memory calculation**:
- Qwen2.5-7B + LoRA + gradients ≈ 7GB base memory
- Each training sample ≈ 2-3GB additional
- T4 (15GB): 7GB + 1×3GB = 10GB (safe)
- L4 (24GB): 7GB + 3×3GB = 16GB (optimal) ⭐
- A100 (40GB): 7GB + 2×3GB = 13GB (plenty of headroom)

### `per_device_eval_batch_size=1` (T4) / `4` (L4) / `3` (A100)
**What it does**: Batch size during validation evaluation.

**GPU-Specific Values**:
```python
# T4: eval_batch_size = 1 (conservative)
# L4: eval_batch_size = 4 (can be higher than train)
# A100: eval_batch_size = 3 (optimized)
```

**Why evaluation can be larger**:
- Evaluation doesn't store gradients (saves memory)
- Forward pass only, less memory intensive
- Can safely use higher batch sizes for faster evaluation

### `gradient_accumulation_steps=8` (T4) / `3` (L4) / `4` (A100)
**What it does**: Accumulate gradients across N steps before updating weights.

**Effective batch size calculation**:
```python
# T4: 1 × 8 = 8 effective batch size
# L4: 3 × 3 = 9 effective batch size
# A100: 2 × 4 = 8 effective batch size
```

**Why 8 effective batch size**:
- **Too small (1-2)**: Noisy gradients, unstable training
- **Optimal (8-16)**: Stable gradients for conversation tasks
- **Too large (32+)**: Slower convergence, less frequent updates

**Memory benefit**: Allows large effective batch size without OOM errors.

---

## Optimization Parameters

### `learning_rate=2e-4`
**What it does**: Step size for weight updates during training.

**Why 2e-4 for LoRA**:
- **Base model fine-tuning**: Usually 1e-5 to 5e-5 (very small)
- **LoRA fine-tuning**: 1e-4 to 5e-4 (higher because only adapter weights trained)
- **Our choice (2e-4)**: Middle ground - fast learning without instability

**LoRA vs Full Fine-tuning**:
```python
# Full fine-tuning: 1e-5 (careful with 7B parameters)
# LoRA: 2e-4 (aggressive with 16M parameters)
```

### `weight_decay=0.01`
**What it does**: L2 regularization to prevent overfitting.

**Why 0.01**:
- **0.0**: No regularization, risk of overfitting
- **0.01**: Standard value, gentle regularization
- **0.1+**: Too aggressive, hurts adaptation

**Effect**: Prevents LoRA adapters from becoming too large, maintains generalization.

### `warmup_ratio=0.03`
**What it does**: Gradually increase learning rate from 0 to target over first 3% of training.

**Why warmup**:
- **Cold start problem**: Starting with full LR can destabilize training
- **LoRA benefit**: Helps adapter weights initialize smoothly
- **3% duration**: 3% of total steps = gentle, not too long

**Calculation example**:
```python
# Total steps: 1000
# Warmup steps: 1000 × 0.03 = 30 steps
# LR schedule: 0 → 2e-4 over first 30 steps
```

### `lr_scheduler_type="cosine"`
**What it does**: Learning rate decay schedule after warmup.

**Why cosine**:
- **Linear**: Sharp drops, can hurt final convergence
- **Cosine**: Smooth decay, good final performance
- **Constant**: No decay, risk of instability

**Cosine shape**:
```
LR │     ╭─╮
   │    ╱   ╲
   │   ╱     ╲
   │  ╱       ╲
   │ ╱         ╲
   │╱           ╲
   └─────────────────► Steps
   Warmup    Cosine Decay
```

---

## Memory Optimization

### `gradient_checkpointing=True`
**What it does**: Trade compute for memory by recomputing activations during backward pass.

**Memory savings**:
- **Without**: Store all activations in memory (~4GB)
- **With**: Recompute on-demand (~1GB memory, +20% compute)

**Why essential for LoRA**: Enables training large models on consumer GPUs.

### `dataloader_pin_memory=False`
**What it does**: Pre-allocate data in GPU memory vs CPU-GPU transfer per batch.

**Why False for Colab**:
- **True**: Faster data transfer, uses more GPU memory
- **False**: Slower transfer, saves precious GPU memory
- **Colab constraint**: Memory is limiting factor, not transfer speed

### `remove_unused_columns=True`
**What it does**: Remove dataset columns not needed for training.

**Memory benefit**:
```python
# Dataset columns: ['conversation', 'metadata', 'source', 'id']
# Training only needs: ['input_ids', 'attention_mask', 'labels']
# Removes: metadata, source, id → saves memory
```

---

## Logging and Evaluation

### `logging_steps=10`
**What it does**: Log training metrics every 10 steps.

**Why every 10 steps**:
- **Too frequent (1-5)**: Spam logs, slow training
- **Good balance (10-20)**: Detailed monitoring without overhead
- **Too sparse (100+)**: Miss important training dynamics

**What gets logged**: Loss, learning rate, training speed, GPU usage.

### `eval_steps=100`
**What it does**: Run validation evaluation every 100 training steps.

**Why 100 steps**:
- **Evaluation is expensive**: Forward pass through validation set
- **Balance**: Regular validation without slowing training
- **Early detection**: Catch overfitting before it's too late

### `evaluation_strategy="steps"`
**What it does**: Evaluate based on step count vs epoch count.

**Why steps vs epochs**:
- **"epoch"**: Evaluate once per epoch (only 3 times total)
- **"steps"**: Evaluate every 100 steps (more frequent monitoring)
- **Better for short training**: 3 epochs = limited evaluation points

---

## Model Saving

### `save_steps=500`
**What it does**: Save model checkpoint every 500 steps.

**Why 500 steps**:
- **Safety**: Protect against crashes/interruptions
- **Disk space**: LoRA adapters are small (~32MB), frequent saves OK
- **Resume capability**: Can restart from recent checkpoint

### `save_total_limit=3`
**What it does**: Keep only the 3 most recent checkpoints.

**Why limit to 3**:
- **Disk space management**: Prevents unlimited checkpoint accumulation
- **Sufficient history**: Keep recent checkpoints for recovery
- **Best model**: Final checkpoint usually best for LoRA

### `load_best_model_at_end=True`
**What it does**: After training, load the checkpoint with best validation score.

**Why True**:
- **Overfitting protection**: Last checkpoint might not be best
- **Automatic selection**: No manual checkpoint comparison needed
- **Quality assurance**: Ensures best performing model is saved

### `metric_for_best_model="eval_loss"`
**What it does**: Use validation loss to determine "best" checkpoint.

**Why eval_loss**:
- **Direct optimization target**: Training minimizes loss
- **Generalization indicator**: Lower validation loss = better generalization
- **Simple and reliable**: Clear metric, no complex evaluation needed

### `greater_is_better=False`
**What it does**: Lower eval_loss is better (not higher).

**Logic**: Loss should decrease, so lower = better performance.

---

## Experiment Tracking

### `run_name=f"marco-lora-{gpu_name}"`
**What it does**: Descriptive name for this training run.

**Naming convention**:
```python
# Examples:
"marco-lora-tesla-t4"
"marco-lora-a100-sxm4-40gb"
```

**Benefits**: Easy identification in logs, wandb dashboard, checkpoint directories.

### `report_to="wandb"`
**What it does**: Send training metrics to Weights & Biases for visualization.

**Why wandb**:
- **Rich visualizations**: Loss curves, learning rate schedules
- **Easy comparison**: Compare different runs side-by-side
- **Remote monitoring**: Check training progress from anywhere
- **Reproducibility**: Stores all hyperparameters automatically

---

## Hardware Optimization

### `fp16=True`
**What it does**: Use 16-bit floating point instead of 32-bit.

**Memory savings**:
- **fp32**: 4 bytes per parameter
- **fp16**: 2 bytes per parameter
- **Total savings**: ~50% memory reduction

**Quality impact**: Minimal for most models, especially with modern GPUs.

**Compatibility**: Works well with T4, A100, and LoRA training.

### `dataloader_num_workers=2`
**What it does**: Use 2 CPU threads for data loading.

**Why 2 workers**:
- **0 workers**: Data loading blocks training (slow)
- **2 workers**: Good parallelism without resource competition
- **4+ workers**: Diminishing returns, potential CPU bottleneck in Colab

---

## Reproducibility

### `seed=self.config.data.data_seed`
**What it does**: Set random seed for reproducible results.

**What gets seeded**:
- PyTorch random number generation
- Model weight initialization
- Data shuffling order
- Dropout patterns

**Why important**:
- **Research validity**: Reproducible experiments
- **Debugging**: Consistent behavior for debugging
- **Comparison**: Fair comparison between different configurations

**Seed value**: Typically 42 (convention) or timestamp-based.

---

## Parameter Interactions and Trade-offs

### Memory vs Speed Trade-offs

```python
# Memory Priority (T4)
batch_size = 1
gradient_accumulation = 8
gradient_checkpointing = True
pin_memory = False

# Speed Priority (A100)
batch_size = 2
gradient_accumulation = 4
gradient_checkpointing = True  # Still worth it
pin_memory = True
```

### Quality vs Training Time

```python
# Quick iteration (development)
epochs = 1
eval_steps = 50
save_steps = 200

# Production quality
epochs = 3-5
eval_steps = 100
save_steps = 500
```

### Stability vs Learning Speed

```python
# Conservative (stable)
learning_rate = 1e-4
warmup_ratio = 0.05
weight_decay = 0.01

# Aggressive (faster learning)
learning_rate = 5e-4
warmup_ratio = 0.01
weight_decay = 0.001
```

---

## Configuration for Different Scenarios

### Quick Debugging Run
```python
num_train_epochs = 1
eval_steps = 10
save_steps = 50
logging_steps = 5
```

### Production Training
```python
num_train_epochs = 3
eval_steps = 100
save_steps = 500
logging_steps = 10
```

### Memory-Constrained Environment
```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
gradient_checkpointing = True
dataloader_pin_memory = False
```

### High-Performance Environment
```python
per_device_train_batch_size = 4
gradient_accumulation_steps = 2
gradient_checkpointing = False
dataloader_pin_memory = True
```

---

## Monitoring and Troubleshooting

### Key Metrics to Watch

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease, stay close to training loss
3. **Learning Rate**: Should follow warmup + decay schedule
4. **GPU Memory**: Should stay under limit with headroom
5. **Training Speed**: Steps per second consistency

### Warning Signs

- **Training loss not decreasing**: LR too low, data issues
- **Validation >> Training loss**: Overfitting, reduce epochs
- **OOM errors**: Reduce batch size, enable optimizations
- **Very slow training**: Increase batch size if memory allows
- **Loss exploding**: LR too high, add more warmup

### Parameter Tuning Guidelines

1. **Start conservative**: Use provided defaults
2. **One change at a time**: Isolate parameter effects
3. **Monitor validation**: Don't overfit to training set
4. **Document changes**: Track what works and what doesn't
5. **Compare fairly**: Same data, same seeds for comparisons

This configuration represents a carefully balanced set of parameters optimized for:
- LoRA fine-tuning of conversational models
- Consumer GPU constraints (T4/A100)
- Italian teaching task requirements
- Stability and reproducibility
- Reasonable training times

The values chosen are based on current research best practices, empirical testing, and the specific constraints of training Marco's Italian teaching capabilities.