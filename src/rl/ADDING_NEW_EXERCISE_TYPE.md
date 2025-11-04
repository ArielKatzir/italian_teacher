# Guide: Adding New Exercise Types

## Example: Adding "Reading Comprehension" Exercise

### ✅ Current System is HIGHLY Extensible!

Your architecture is well-designed for adding new exercise types. Here's what you need to do:

---

## Step 1: Update Exercise Type List (generate_training_requests.py)

```python
# Line 50 in generate_training_requests.py
EXERCISE_TYPES = [
    "fill_in_blank",
    "multiple_choice",
    "translation",
    "reading_comprehension"  # ✅ ADD NEW TYPE
]
```

**That's it for the generator!** The system will automatically include it in training requests.

---

## Step 2: Update Reward Function Scorers (if needed)

Most scorers are **already compatible** with new types! Here's what works out-of-the-box:

### ✅ **Auto-Compatible Scorers** (8/8):

1. **JSONScorer**: ✅ Works (validates structure)
2. **ExerciseQualityScorer**: ⚠️ Needs small update (see below)
3. **LinguisticScorer**: ✅ Works (extracts Italian text)
4. **CEFRScorer**: ✅ Works (checks difficulty)
5. **FluencyScorer**: ✅ Works (checks naturalness)
6. **GrammarScorer**: ✅ Works (validates grammar focus)
7. **CoherenceScorer**: ✅ Works (checks logical sense)
8. **TopicScorer**: ✅ Works (checks topic relevance)

---

## Step 3: Update ExerciseQualityScorer for New Type

**File:** `src/rl/reward_function/scorers/exercise_quality_scorer.py`

**Current code** (lines 46-115) handles:
- `fill_in_blank` → checks for hint
- `translation` → checks for valid structure

**Add reading_comprehension handling:**

```python
# Add after line 114 (after translation check)
elif exercise_type == "reading_comprehension" and question and answer:
    # Validate reading comprehension structure
    is_valid_structure = True

    # Check 1: Question should contain a passage (longer text)
    if len(question.split()) < 30:  # Passage should be substantial
        is_valid_structure = False
        errors.append("CRITICAL: Reading comprehension passage too short (< 30 words).")

    # Check 2: Answer should be a reasonable response
    if len(answer.strip()) < 3:
        is_valid_structure = False
        errors.append("CRITICAL: Reading comprehension answer too short.")

    # Check 3: Optionally check if passage is in Italian
    italian_indicators = {"il", "la", "un", "una", "di", "a", "è", "sono", "ho", "ha"}
    passage_words = set(re.findall(r'\b[a-zA-Z]+\b', question.lower()))
    if not any(word in italian_indicators for word in passage_words):
        is_valid_structure = False
        errors.append("CRITICAL: Reading comprehension passage does not appear to be Italian.")

    if not is_valid_structure:
        context_score = 0.0

score += context_score - 10.0
```

---

## Step 4: Update Text Extraction Helpers (optional)

**File:** `src/rl/reward_function/scorers/text_utils.py`

Add reading comprehension case to `extract_italian_text`:

```python
# Add after line 32
# For reading_comprehension, extract passage + answer
if ex_type == "reading_comprehension":
    passage = question.strip()
    answer_text = answer.strip()
    return f"{passage} {answer_text}"
```

---

## Step 5: Test the New Type

```python
# Add to test_reward_function.py
{
    "name": "READING COMPREHENSION: Basic test",
    "exercise": {
        "type": "reading_comprehension",
        "question": "Maria va al mercato ogni sabato. Lei compra frutta e verdura fresca. Le piace parlare con i venditori. Oggi ha comprato mele, banane e pomodori. È molto contenta.\n\nDove va Maria ogni sabato?",
        "correct_answer": "Al mercato",
        "explanation": "Comprehension of simple present tense passage"
    },
    "request": {
        "level": "A2",
        "grammar_focus": "present_tense",
        "topic": "fare la spesa",
        "num_exercises": 1,
        "exercise_types": ["reading_comprehension"]
    },
    "expected_range": (70, 90)
}
```

---

## Step 6: Retrain the Model

Run your training script with the new exercise type included in requests. The model will learn to generate it!

```python
# Your training will automatically include the new type in requests
requests = generate_training_requests(num_requests=800)
# Some requests will now include "reading_comprehension"
```

---

## 📊 Summary: Effort Required

| Task | Effort | Required? |
|------|--------|-----------|
| Add to EXERCISE_TYPES list | 1 line | ✅ Yes |
| Update ExerciseQualityScorer | 20 lines | ✅ Yes |
| Update text extraction | 5 lines | 🟡 Optional |
| Add test case | 15 lines | ✅ Yes (recommended) |
| Retrain model | Full training run | ✅ Yes |

**Total Dev Time:** ~30 minutes of coding + training time

---

## ✅ Your System is Extensible!

**Pros:**
- Most scorers work automatically with new types
- Clear extension points (ExerciseQualityScorer, text_utils)
- No changes needed to GRPO trainer or data pipeline

**Cons:**
- Need to manually add validation logic for each new type
- Need to retrain model (expected for any new capability)

---

## 🚀 Recommendations:

1. **Keep current 3 types for now** (fill_in_blank, multiple_choice, translation)
   - Well-tested and working

2. **Add reading_comprehension in Round 3** (after this training)
   - Follow steps above
   - Start with simple passages (A1/A2)

3. **Future types to consider:**
   - `sentence_ordering` (put words/phrases in correct order)
   - `error_correction` (find and fix the mistake)
   - `dialogue_completion` (fill in missing dialogue lines)
   - `listening_comprehension` (transcription + questions, requires audio)

---

## Example JSON Schema for Reading Comprehension:

```json
{
  "type": "reading_comprehension",
  "question": "Passage text here...\n\nQuestion about the passage?",
  "correct_answer": "The answer",
  "options": ["answer1", "answer2", "answer3", "correct"],  // Optional for MC format
  "explanation": "Explanation of why this is correct"
}
```

The model would learn to generate this structure from your training examples!
