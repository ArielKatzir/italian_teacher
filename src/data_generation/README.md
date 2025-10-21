# Data Generation & Augmentation

Scripts for generating and augmenting the Italian exercise training dataset.

## augment_vocabulary.py

Generates 500 new training examples using GPT-4o-mini to address vocabulary and topic gaps.

### Purpose

Fixes issues identified in the current model (alpha=8):
- **Gender errors**: "gli aquile" → "le aquile" (missing vocabulary)
- **Topic drift**: Limited vocabulary leads to confusion
- **Tense mismatches**: Not enough past_tense examples in training

### What it generates

**Priority 1: Diverse Vocabulary** (new words not in training data)
- Animals: ragno, lombrico, aquila, serpente, farfalla, formica, ape, vespa, etc.
- Nature: fiore, albero, bosco, fiume, montagna, lago, cascata, etc.
- Food: zucca, melanzana, carciofo, asparago, cavolo, etc.
- Professions: architetto, veterinario, scienziato, ingegnere, etc.

**Priority 2: Balanced Grammar Focus**
- past_tense: +150 examples (currently only 124)
- present_tense: +100 examples (boost to 250+)
- reflexive_verbs: +50 examples
- Other focuses maintained proportionally

**Priority 4: Topic Diversity**
- animals: 150 examples
- nature_environment: 100 examples
- daily_life: 100 examples
- food_cooking: 75 examples
- professions_work: 75 examples

### Usage

1. **Set OpenAI API key:**
```bash
export OPENAI_API_KEY='your-key-here'
```

2. **Run the script:**
```bash
python src/data_generation/augment_vocabulary.py
```

3. **Review generated examples:**
```bash
head -5 data/datasets/v4_augmented/train_augmentation.jsonl | jq
```

4. **Check vocabulary coverage:**
```bash
# Check if "ragno" is now present
grep -c "ragno" data/datasets/v4_augmented/train_augmentation.jsonl

# Check if "lombrico" is now present
grep -c "lombrico" data/datasets/v4_augmented/train_augmentation.jsonl

# Check if "aquila" is now present
grep -c "aquila" data/datasets/v4_augmented/train_augmentation.jsonl
```

5. **Merge with existing training data:**
```bash
# Create augmented directory
mkdir -p data/datasets/v4_augmented

# Merge datasets
cat data/datasets/final/train.jsonl \
    data/datasets/v4_augmented/train_augmentation.jsonl \
    > data/datasets/v4_augmented/train.jsonl

# Copy validation and test unchanged
cp data/datasets/final/validation.jsonl data/datasets/v4_augmented/
cp data/datasets/final/test.jsonl data/datasets/v4_augmented/

# Verify counts
echo "Original: $(wc -l < data/datasets/final/train.jsonl)"
echo "Augmented: $(wc -l < data/datasets/v4_augmented/train_augmentation.jsonl)"
echo "Total: $(wc -l < data/datasets/v4_augmented/train.jsonl)"
```

6. **Retrain with V4 config:**
Update `src/fine_tuning/config_exercise_generation.py` to point to new dataset:
```python
train_file: str = "data/datasets/v4_augmented/train.jsonl"
```

### Output Format

Each generated example follows the same format as existing training data:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert Italian language teacher..."
    },
    {
      "role": "user",
      "content": "Generate 3 exercises:\nCEFR Level: A2\nGrammar Focus: past_tense\nTopic: animals\n..."
    },
    {
      "role": "assistant",
      "content": "[{\"type\": \"fill_in_blank\", \"question\": \"Il ragno ___ (salire) sul muro.\", \"answer\": \"è salito\", ...}]"
    }
  ],
  "metadata": {
    "cefr_level": "A2",
    "topic": "animals",
    "num_exercises": 3,
    "grammar_focus": "past_tense",
    "source": "gpt4o_mini_augmented",
    "vocabulary_focus": "ragno, aquila, lombrico"
  }
}
```

### Expected Results

After retraining with V4 config (alpha=6) and augmented data:

**Before (alpha=8, 3,186 examples):**
- ❌ "Gli aquile" (gender error - missing vocabulary)
- ⚠️  Translation in present tense when grammar_focus=past_tense
- 🟠 Limited vocabulary coverage (~10,345 words)

**After (alpha=6, 3,686 examples):**
- ✅ "Le aquile" (correct gender - vocabulary in training data)
- ✅ All exercises match grammar focus tense
- ✅ Wide vocabulary coverage (~15,000+ words)
- ✅ Better base model knowledge preservation

### Technical Details

**Model**: gpt-4o-mini
**Temperature**: 0.7 (same as original dataset generation)
**Max Tokens**: 2000
**Validation**: JSON parsing, field checks

**Why GPT-4o-mini?**
- Successfully generated 44.6% of existing dataset (1,776 examples)
- Fast and cost-effective
- Good Italian language quality
- Proven track record from dataset v1.0

### Cost Estimation

**GPT-4o-mini pricing** (as of 2025):
- Input: $0.15 / 1M tokens
- Output: $0.60 / 1M tokens

**Estimated cost for 500 examples:**
- Input: ~500 prompts × 300 tokens = 150K tokens = $0.02
- Output: ~500 responses × 800 tokens = 400K tokens = $0.24
- **Total: ~$0.26** (less than 30 cents)

### Troubleshooting

**API key not set:**
```bash
export OPENAI_API_KEY='your-key-here'
```

**JSON parsing errors:**
The script automatically strips markdown code blocks and validates JSON.
If errors persist, check OpenAI API response format.

**Rate limits:**
Add delay between requests:
```python
import time
time.sleep(0.5)  # Wait 500ms between requests
```

## Next Steps

After generating and merging the augmented dataset:

1. **Review quality**: Sample 10-20 examples manually
2. **Update config**: Point training to `v4_augmented/train.jsonl`
3. **Retrain**: Use V4 config (alpha=6, rank=12, 2 modules)
4. **Validate**: Test with "ragno", "lombrico", "aquila" examples
5. **Compare**: Check if gender errors and tense mismatches are fixed
