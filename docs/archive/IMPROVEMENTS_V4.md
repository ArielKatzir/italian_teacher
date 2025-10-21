# V4 Improvements: Preventing Catastrophic Forgetting

**Date**: October 10, 2025
**Goal**: Fix gender errors, tense mismatches, and topic drift by preserving base model Italian knowledge

## Problem Analysis

### Issues with V3 (alpha=8, rank=12)
- ❌ **Gender errors**: "Gli aquile" → should be "Le aquile" (feminine)
- 🟠 **Tense mismatches**: Translation exercise in present tense when grammar_focus=past_tense
- ⚠️ **Missing vocabulary**: "ragno" (0 occurrences), "lombrico" (0), "aquila" (0) in training data

### Root Cause
**Catastrophic forgetting** due to strong LoRA adaptation:
- Alpha=8 (0.67× rank) overwrites base model knowledge
- Base LLaMAntino knows 50,000+ Italian words correctly
- Training data only has ~10,345 unique words
- Out-of-vocabulary words get wrong gender/grammar

## V4 Solutions Implemented

### 1. ✅ Weaker LoRA Configuration

**File**: [src/fine_tuning/config_exercise_generation.py](src/fine_tuning/config_exercise_generation.py)

**Changes**:
```python
# V3 (too strong, causes forgetting)
r: int = 12
lora_alpha: int = 8  # 0.67× rank
lora_dropout: float = 0.12
target_modules: ['v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']  # 5 modules

# V4 (weaker, preserves base knowledge)
r: int = 12
lora_alpha: int = 6  # 0.5× rank - WEAKER
lora_dropout: float = 0.15  # Higher dropout
target_modules: ['q_proj', 'v_proj']  # Only 2 modules
```

**Rationale**:
- Lower alpha preserves more base model weights
- Fewer target modules = less interference with base knowledge
- Higher dropout adds regularization

### 2. ✅ Enhanced Prompt with Tense Enforcement

**File**: [src/api/inference/colab_api.py:113-122](src/api/inference/colab_api.py#L113-L122)

**Changes**:
```python
CRITICAL RULES:
3. GRAMMAR: Every exercise MUST test "{grammer}" at {request.cefr_level} level
   - For "past_tense": Use passato prossimo (ho fatto, sono andato) or imperfetto
   - For "present_tense": Use presente indicativo (faccio, vado)
   - For "future_tense": Use futuro semplice (farò, andrò)
   - ALL exercises must use the same tense as the grammar focus
```

**Impact**: Clear tense expectations reduce model confusion

### 3. ✅ Post-Generation Validation

**File**: [src/api/inference/colab_api.py:446-634](src/api/inference/colab_api.py#L446-L634)

**A. Enhanced Gender Validation**:
```python
def _get_correct_article(article: str, gender: str) -> Optional[str]:
    # Now handles plural: gli/le/i
    if gender == 'Masc':
        if article_lower == 'le':  # Plural feminine with masculine noun
            return 'i'  # or 'gli'
    elif gender == 'Fem':
        if article_lower in ['i', 'gli']:  # Plural masculine with feminine noun
            return 'le'
```

**B. New Tense Consistency Checker**:
```python
def _check_tense_consistency(doc, grammar_focus: str, exercise_idx: int, field: str):
    expected_tenses = {
        'past_tense': {'Past'},
        'present_tense': {'Pres'},
        'future_tense': {'Fut'},
    }

    # Find verbs and check tense
    verbs = [token for token in doc if token.pos_ == 'VERB']

    # Report mismatches
    if found_tenses and not found_tenses.intersection(expected):
        print(f"🟠 Exercise {exercise_idx + 1} ({field}): Tense mismatch")
```

**Impact**: Catches validation issues before returning to user

### 4. ✅ Vocabulary Augmentation Script

**File**: [src/data_generation/augment_vocabulary.py](src/data_generation/augment_vocabulary.py)

**Purpose**: Generate 500 new examples with missing vocabulary using GPT-4o-mini

**Target vocabulary** (Priority 1):
- Animals: ragno, lombrico, aquila, serpente, farfalla, formica, ape, vespa...
- Nature: fiore, albero, bosco, fiume, montagna, lago, cascata...
- Food: zucca, melanzana, carciofo, asparago, cavolo, fungo...
- Professions: architetto, veterinario, scienziato, ingegnere...

**Grammar balance** (Priority 2):
- past_tense: +149 examples (currently 124)
- present_tense: +100 examples (currently 149)
- reflexive_verbs: +49 examples (currently 94)

**Topic diversity** (Priority 4):
- animals: 150 examples
- nature_environment: 100 examples
- daily_life: 100 examples
- food_cooking: 72 examples
- professions_work: 72 examples

**Verified**: Test shows all critical words (ragno, lombrico, aquila) are covered ✅

## Implementation Status

| Task | Status | File |
|------|--------|------|
| Lower LoRA alpha to 6 | ✅ Done | config_exercise_generation.py |
| Reduce target modules to 2 | ✅ Done | config_exercise_generation.py |
| Add tense rules to prompt | ✅ Done | colab_api.py |
| Add tense validation | ✅ Done | colab_api.py |
| Strengthen gender validation | ✅ Done | colab_api.py |
| Create augmentation script | ✅ Done | augment_vocabulary.py |
| Test augmentation script | ✅ Passed | test_augment.py |
| Generate 500 new examples | ⏳ Pending | Run script with API key |
| Merge augmented dataset | ⏳ Pending | After generation |
| Retrain with V4 config | ⏳ Pending | After merge |

## Next Steps

### Step 1: Generate Augmented Data
```bash
# Set OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Run generation (cost: ~$0.26)
python src/data_generation/augment_vocabulary.py

# Expected output: data/datasets/v4_augmented/train_augmentation.jsonl
```

### Step 2: Verify Quality
```bash
# Check vocabulary coverage
grep -c "ragno" data/datasets/v4_augmented/train_augmentation.jsonl
grep -c "lombrico" data/datasets/v4_augmented/train_augmentation.jsonl
grep -c "aquila" data/datasets/v4_augmented/train_augmentation.jsonl

# Sample exercises
head -5 data/datasets/v4_augmented/train_augmentation.jsonl | jq '.messages[-1].content' | jq
```

### Step 3: Merge Datasets
```bash
# Create merged training set
cat data/datasets/final/train.jsonl \
    data/datasets/v4_augmented/train_augmentation.jsonl \
    > data/datasets/v4_augmented/train.jsonl

# Copy validation/test unchanged
cp data/datasets/final/validation.jsonl data/datasets/v4_augmented/
cp data/datasets/final/test.jsonl data/datasets/v4_augmented/

# Verify
echo "Original: $(wc -l < data/datasets/final/train.jsonl)"
echo "New: $(wc -l < data/datasets/v4_augmented/train_augmentation.jsonl)"
echo "Total: $(wc -l < data/datasets/v4_augmented/train.jsonl)"
```

### Step 4: Update Config
```python
# In src/fine_tuning/config_exercise_generation.py
train_file: str = "data/datasets/v4_augmented/train.jsonl"
```

### Step 5: Retrain Model
```bash
# Upload to Colab and run training notebook
# Expected training time: ~2-3 hours on L4/A100
# Output: models/italian_exercise_generator_lora_v4
```

### Step 6: Test V4 Model
```bash
# Test with problematic examples from alpha=8
./teacher homework create --level A2 --grammar past_tense --topic eagles --exercises 3

# Expected improvements:
# ✅ "Le aquile" (not "Gli aquile")
# ✅ All exercises in past tense
# ✅ No topic drift
```

## Expected Results

### Before V4 (alpha=8, 3,186 examples)
```
Exercise 2: Translation
Question: The eagles migrate to warmer regions in winter.
Answer: Le aquile migrano verso regioni più calde in inverno.
Issue: 🟠 Present tense, but grammar_focus=past_tense
```

### After V4 (alpha=6, 3,686 examples)
```
Exercise 2: Translation
Question: The eagles migrated to warmer regions last winter.
Answer: Le aquile sono migrate verso regioni più calde l'inverno scorso.
✅ Correct: Past tense matches grammar_focus
✅ Correct: "Le aquile" (feminine plural)
```

## Technical Justification

### Why alpha=6 instead of 8?
- **Standard LoRA**: alpha = 2× rank (would be 24 for rank=12)
- **V3**: alpha = 0.67× rank = 8 (too strong)
- **V4**: alpha = 0.5× rank = 6 (weaker, preserves base)

Research shows:
- Lower alpha → Less catastrophic forgetting
- Base model knowledge preserved for out-of-vocabulary words
- Task-specific behavior still learned (JSON format, exercise structure)

### Why only 2 target modules?
- q_proj: Query projection (attention steering)
- v_proj: Value projection (content transformation)

Removed:
- k_proj, o_proj: Keep base attention mechanism intact
- gate_proj, up_proj, down_proj: Keep base MLP intact

Result: LoRA learns task structure but doesn't override Italian grammar

### Why GPT-4o-mini for augmentation?
- ✅ Successfully generated 44.6% of existing dataset (1,776 examples)
- ✅ Cost-effective (~$0.26 for 500 examples)
- ✅ Good Italian quality
- ✅ Fast generation

## Cost Analysis

**Data Generation**:
- GPT-4o-mini: ~$0.26 for 500 examples
- Total: < $0.30

**Training**:
- Colab L4 GPU: Free tier or ~$0.50/hour
- Training time: 2-3 hours
- Total: Free or ~$1.50

**Total V4 cost**: < $2.00

## Success Metrics

After V4 deployment, measure:

1. **Gender accuracy**: % exercises with correct article-noun agreement
2. **Tense consistency**: % exercises matching grammar_focus tense
3. **Vocabulary coverage**: Can model handle "ragno", "lombrico", "aquila"?
4. **Topic adherence**: % exercises staying on topic
5. **Overall quality**: User satisfaction, no grammar errors

Target: 95%+ accuracy on all metrics

## References

- [config_exercise_generation.py](src/fine_tuning/config_exercise_generation.py) - LoRA config
- [colab_api.py](src/api/inference/colab_api.py) - Inference + validation
- [augment_vocabulary.py](src/data_generation/augment_vocabulary.py) - Data generation
- [README.md](src/data_generation/README.md) - Usage instructions
