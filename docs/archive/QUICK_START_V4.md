# Quick Start: V4 Training Pipeline

Complete commands to go from current state to V4 trained model.

## Summary

**What changed**:
1. ✅ LoRA alpha lowered from 8→6 (weaker adaptation, preserves base knowledge)
2. ✅ Target modules reduced from 5→2 (less interference)
3. ✅ Enhanced prompts with tense enforcement
4. ✅ spaCy validation for gender + tense checking
5. ✅ Augmentation script for 500 new vocabulary-rich examples

**Why**: Fix "Gli aquile"→"Le aquile", tense mismatches, vocabulary gaps

**Cost**: ~$0.26 for data generation (GPT-4o-mini)

---

## Step-by-Step Commands

### 1. Generate Augmented Data (~5-10 minutes)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='sk-...'

# Run generation script
cd "/Users/arielkatzir/Library/CloudStorage/GoogleDrive-ari.katzir@gmail.com/My Drive/Colab Notebooks/italian_teacher"
~/.venvs/py312/bin/python src/data_generation/augment_vocabulary.py
```

**Expected output**:
```
🚀 Italian Exercise Vocabulary Augmentation
   Target: 500 new examples
   Output: data/datasets/v4_augmented/train_augmentation.jsonl

📝 Generating prompt specifications...
   Created 494 prompts

🤖 Generating exercises with GPT-4o-mini...
   [1/494] A2 - animals - past_tense ✅
   [2/494] B1 - nature environment - present_tense ✅
   ...

✅ Generated 494 examples
💾 Saving to data/datasets/v4_augmented/train_augmentation.jsonl...
```

### 2. Verify Generated Data

```bash
# Check file was created
ls -lh data/datasets/v4_augmented/train_augmentation.jsonl

# Count lines
wc -l data/datasets/v4_augmented/train_augmentation.jsonl

# Check vocabulary coverage
echo "ragno: $(grep -c 'ragno' data/datasets/v4_augmented/train_augmentation.jsonl)"
echo "lombrico: $(grep -c 'lombrico' data/datasets/v4_augmented/train_augmentation.jsonl)"
echo "aquila: $(grep -c 'aquila' data/datasets/v4_augmented/train_augmentation.jsonl)"

# Sample one example
head -1 data/datasets/v4_augmented/train_augmentation.jsonl | jq '.'
```

**Expected vocabulary counts**:
- ragno: 10-20 occurrences (was 0)
- lombrico: 5-15 occurrences (was 0)
- aquila: 15-25 occurrences (was 0)

### 3. Merge with Original Dataset

```bash
# Merge training data
cat data/datasets/final/train.jsonl \
    data/datasets/v4_augmented/train_augmentation.jsonl \
    > data/datasets/v4_augmented/train.jsonl

# Copy validation and test sets unchanged
cp data/datasets/final/validation.jsonl data/datasets/v4_augmented/validation.jsonl
cp data/datasets/final/test.jsonl data/datasets/v4_augmented/test.jsonl

# Verify counts
echo "✅ Dataset sizes:"
echo "   Original train: $(wc -l < data/datasets/final/train.jsonl) examples"
echo "   Augmented train: $(wc -l < data/datasets/v4_augmented/train_augmentation.jsonl) examples"
echo "   Merged train: $(wc -l < data/datasets/v4_augmented/train.jsonl) examples"
echo "   Validation: $(wc -l < data/datasets/v4_augmented/validation.jsonl) examples"
echo "   Test: $(wc -l < data/datasets/v4_augmented/test.jsonl) examples"
```

**Expected output**:
```
✅ Dataset sizes:
   Original train: 3186 examples
   Augmented train: 494 examples
   Merged train: 3680 examples
   Validation: 394 examples
   Test: 403 examples
```

### 4. Update Training Config

Edit [src/fine_tuning/config_exercise_generation.py](src/fine_tuning/config_exercise_generation.py):

```python
# Change line 95-100:
train_file: str = (
    "/content/drive/MyDrive/Colab Notebooks/italian_teacher/data/datasets/v4_augmented/train.jsonl"
)
validation_file: str = (
    "/content/drive/MyDrive/Colab Notebooks/italian_teacher/data/datasets/v4_augmented/validation.jsonl"
)
test_file: str = (
    "/content/drive/MyDrive/Colab Notebooks/italian_teacher/data/datasets/v4_augmented/test.jsonl"
)
```

**Already done**:
- ✅ LoRA alpha changed to 6 (line 18)
- ✅ Target modules reduced to 2 (line 30-31)
- ✅ Dropout increased to 0.15 (line 20)

### 5. Train Model on Colab

**Option A**: Use existing training notebook
```bash
# Open the training notebook (if exists)
# Update dataset paths as above
# Run all cells
```

**Option B**: Run training script directly
```python
# In Colab with GPU (L4 or A100)
!git clone <your-repo>
!cd italian_teacher && pip install -r requirements.txt

# Run training
!python src/fine_tuning/train_exercise_generator.py
```

**Training time**:
- L4 GPU: ~3-4 hours
- A100 GPU: ~2-3 hours

**Output**:
- Model saved to: `models/italian_exercise_generator_lora_low_alpha/`

### 6. Update API to Use V4 Model

Edit [demos/colab_inference_api.ipynb](demos/colab_inference_api.ipynb):

```python
# Cell 3: Update model path
LORA_PATH = os.path.join(PROJECT_ROOT, "models/italian_exercise_generator_lora_low_alpha")
```

Or if using merged model:
```python
# Cell 4: Update merged path
MERGED_MODEL_PATH = "/content/italian_teacher_model_v4_merged"
```

### 7. Test V4 Model

```bash
# Restart API with new model
# Then test with problematic examples:

./teacher homework create \
  --level A2 \
  --grammar past_tense \
  --topic eagles \
  --exercises 3

./teacher homework create \
  --level A2 \
  --grammar past_tense \
  --topic spiders \
  --exercises 3

./teacher homework create \
  --level A2 \
  --grammar past_tense \
  --topic worms \
  --exercises 3
```

**Expected improvements**:
- ✅ "Le aquile" not "Gli aquile"
- ✅ "Il ragno" not "La ragno"
- ✅ All exercises in past tense
- ✅ No topic drift

---

## Troubleshooting

### OpenAI API key not working
```bash
# Check key is set
echo $OPENAI_API_KEY

# Test API
~/.venvs/py312/bin/python -c "from openai import OpenAI; client = OpenAI(); print('✅ API key works')"
```

### Not enough augmented examples generated
```bash
# Check how many failed
grep "❌" <generation_log>

# Regenerate only failed ones
# Or adjust TARGET_EXAMPLES in augment_vocabulary.py and re-run
```

### Training OOM (out of memory)
```python
# In config_exercise_generation.py, reduce batch size:
per_device_train_batch_size: int = 1  # Was 4
gradient_accumulation_steps: int = 16  # Was 4
```

### Model still makes errors
- Check augmented data quality: `jq` sample examples
- Verify LoRA config: alpha should be 6, not 8
- Consider lowering alpha further to 4

---

## Files Changed

| File | Change | Status |
|------|--------|--------|
| src/fine_tuning/config_exercise_generation.py | alpha→6, modules→2 | ✅ |
| src/api/inference/colab_api.py | Tense validation, gender fix | ✅ |
| src/data_generation/augment_vocabulary.py | New script | ✅ |
| data/datasets/v4_augmented/train.jsonl | Merged dataset | ⏳ |
| models/italian_exercise_generator_lora_low_alpha/ | Trained model | ⏳ |

---

## Quick Reference

**Generate data**: `python src/data_generation/augment_vocabulary.py`

**Test script**: `python src/data_generation/test_augment.py`

**Merge data**: `cat final/train.jsonl v4_augmented/train_augmentation.jsonl > v4_augmented/train.jsonl`

**Check vocab**: `grep -c "ragno" data/datasets/v4_augmented/train.jsonl`

**Train**: Run Colab training notebook with updated paths

**Test**: `./teacher homework create --level A2 --grammar past_tense --topic eagles`

---

## Success Criteria

V4 is successful if:
- ✅ Gender accuracy >95% (including "aquila", "ragno", "lombrico")
- ✅ Tense consistency 100% (exercises match grammar_focus)
- ✅ No topic drift
- ✅ Realistic, natural Italian sentences
- ✅ 4 distinct multiple choice options

Compare V3 vs V4 with same test prompts to measure improvement.
