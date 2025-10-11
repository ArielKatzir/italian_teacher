# ⚠️ MUST TRAIN NEW MODEL

## Current Situation

**You tested the OLD model** - that's why results are bad!

```
Current API model:  models/italian_exercise_generator_v4
├─ Trained on:      3,186 examples (OLD dataset)
├─ Alpha:           8 (too strong)
├─ Target modules:  5 modules
└─ Result:          ❌ Gender errors, tense errors, bad quality
```

**But you prepared NEW training data:**

```
New V4 dataset:     data/datasets/v4_augmented/train.jsonl
├─ Examples:        8,859 (2.7× larger)
├─ Alpha:           6 (weaker - config updated)
├─ Target modules:  2 modules (config updated)
└─ Status:          ✅ Ready but NOT TRAINED YET
```

---

## What Went Wrong

Looking at your test output:
```
❌ Exercise 1: nuotavano (imperfect) - acceptable for past_tense
❌ Exercise 2: "migrano" (present) - WRONG, should be past tense
❌ Exercise 3: "mangiano" (present) - WRONG, should be past tense
❌ Exercise 5: "Gli ottoni" - WRONG translation (should be "I polpi")
❌ Exercise 6-9: All present tense - WRONG
```

**This is the OLD model behavior** - exactly what we're trying to fix!

---

## Training Data is GOOD

I checked the new dataset - it's excellent:

```python
# Sample from v4_augmented/train.jsonl with grammar_focus=past_tense:

{
  "question": "La settimana scorsa ___ (comprare) un libro nuovo.",
  "answer": "ho comprato/a",  # ✅ Correct past tense
  "grammar_focus": "past_tense"
}

{
  "question": "L'anno scorsa, Maria ____ una maratona",
  "answer": "ha corso",  # ✅ Correct past tense
  "grammar_focus": "past_tense"
}

{
  "question": "Io (andare) in vacanza a Roma l'anno scorso.",
  "answer": "sono andato/a",  # ✅ Correct past tense
  "grammar_focus": "past_tense"
}
```

All training examples use CORRECT past tense forms!

---

## What You Need To Do

### Step 1: Train NEW Model on Colab

Open your training notebook and run with these settings:

```python
# Dataset (already configured in config_exercise_generation.py)
train_file = "data/datasets/v4_augmented/train.jsonl"  # ✅ 8,859 examples

# LoRA Config (already configured)
lora_alpha = 6  # ✅ Weaker
target_modules = ['q_proj', 'v_proj']  # ✅ Only 2

# Output path
output_dir = "models/italian_exercise_generator_lora_low_alpha"
```

**Training time**: 2-5 hours (depending on GPU)

### Step 2: Update API Notebook

After training completes, update `demos/colab_inference_api.ipynb` Cell 3:

```python
# OLD (current):
LORA_PATH = os.path.join(PROJECT_ROOT, "models/italian_exercise_generator_v4")

# NEW (after training):
LORA_PATH = os.path.join(PROJECT_ROOT, "models/italian_exercise_generator_lora_low_alpha")
```

### Step 3: Restart API & Test

```bash
# In Colab: Restart and run all cells
# Then test again:
./teacher assignment create -s 1 -l A1 -t sea_animals -g past_tense -q 10
./student homework view -h 6 -s 1
```

**Expected results after V4 training:**
- ✅ All exercises in past tense (nuotavano, hanno nuotato, sono migrati)
- ✅ Correct translations (I polpi, not "Gli ottoni")
- ✅ No tense mismatches
- ✅ Better quality overall

---

## Why Training Will Fix This

**OLD model (what you just tested):**
```
3,186 examples + alpha=8 + 5 modules
= Strong adaptation
= Catastrophic forgetting
= Wrong tenses, wrong translations
```

**NEW model (what you'll train):**
```
8,859 examples + alpha=6 + 2 modules
= Weak adaptation
= Preserves base knowledge
= Correct Italian grammar
```

The comprehensive dataset (8,859 examples) with weaker LoRA (alpha=6) will preserve the base model's Italian knowledge while learning the exercise generation task.

---

## Current Model Path Issue

Your API notebook shows:
```python
LORA_PATH = "models/italian_exercise_generator_v4"
# This points to OLD model!
```

But your config says:
```python
output_dir = "models/italian_exercise_generator_lora_low_alpha"
# This is where NEW model will be saved
```

After training, you'll have TWO models:
```
models/
├── italian_exercise_generator_v4/  (OLD - alpha=8, 3K examples)
└── italian_exercise_generator_lora_low_alpha/  (NEW - alpha=6, 8.8K examples)
```

Make sure to use the NEW one!

---

## Summary

✅ Dataset is ready (8,859 examples, excellent quality)
✅ Config is ready (alpha=6, 2 modules)
❌ Model NOT trained yet
❌ API using OLD model

**Next action**: Train the NEW model on Colab, then update API to use it.

The bad results you saw are expected - you're testing the old model that has all the problems we're trying to fix!
