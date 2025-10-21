# Modular Reward Function for Italian Exercise Generation

## Overview

This modular reward function scores Italian exercises on **6 dimensions** (0-100 points total):

1. **JSON Validity** (15 points) - Structure and format validation
2. **Linguistic Quality** (35 points) - Comprehensive Italian grammar validation
3. **CEFR Level Alignment** (20 points) - Appropriate difficulty for target level
4. **Fluency** (10 points) - Natural language flow and construction
5. **Grammar Correctness** (10 points) - Matches requested grammar_focus
6. **Topic Adherence** (10 points) - Relevant to requested topic

## Architecture

### Modular Design

The reward function is organized into:

```
src/rl/
├── data/                           # Data modules (16,887-word vocabulary + rules)
│   ├── __init__.py
│   ├── vocabulary_lists.py         # Italian vocabulary with CEFR tags
│   ├── gender_exceptions.py        # 60+ gender exception words
│   ├── article_rules.py            # Comprehensive article rules
│   ├── invariant_adjectives.py     # Invariant adjective list
│   ├── cefr_rules.py              # CEFR level complexity rules
│   └── italian_vocabulary.json     # 16,887 words from Language-Learning-decks
│
├── reward_function/                # Modular reward function
│   ├── __init__.py
│   ├── reward_function_modular.py  # Main coordinator class
│   ├── scorers/                    # Individual scorer components
│   │   ├── __init__.py
│   │   ├── base.py                # BaseScorer abstract class
│   │   ├── json_scorer.py         # JSON structure validation
│   │   ├── linguistic_scorer.py   # Italian grammar validation
│   │   ├── cefr_scorer.py         # CEFR level alignment
│   │   ├── fluency_scorer.py      # Fluency and naturalness
│   │   ├── grammar_scorer.py      # Grammar focus validation
│   │   └── topic_scorer.py        # Semantic topic similarity
│   └── README.md                  # This file
│
└── reward_function.py              # Legacy monolithic version (deprecated)
```

### Key Design Principles

1. **Separation of Concerns**: Each scorer is responsible for one aspect
2. **Data Modularity**: All data (vocabulary, rules, exceptions) in separate files
3. **Easy Extensibility**: Add new scorers by inheriting from `BaseScorer`
4. **Performance**: Pre-loads and caches vocabulary (16,887 words) at initialization
5. **Type Safety**: Uses dataclasses and type hints throughout

## Comprehensive Italian Vocabulary

The CEFR scorer now uses **16,887 authentic Italian words** with CEFR level tags from the [Language-Learning-decks](https://github.com/vbvss199/Language-Learning-decks) repository.

### Vocabulary Statistics

- **Total words**: 16,887
- **By CEFR level**:
  - A1: 626 words
  - A2: 1,884 words
  - B1: 5,094 words
  - B2: 6,649 words
  - C1: 2,597 words
  - C2: 37 words
- **By part of speech**:
  - Nouns: 9,956
  - Adjectives: 4,195
  - Verbs: 2,079
  - Adverbs: 653

### Cumulative Vocabulary

Vocabulary is cumulative by default: B1 students know A1+A2+B1 words (7,604 total).

## Usage

### Basic Usage

```python
from src.rl.reward_function import ExerciseRewardFunction

# Initialize reward function (loads models and vocabulary)
rf = ExerciseRewardFunction()

# Score an exercise
exercise = {
    "type": "fill_in_blank",
    "question": "Io ___ la pizza.",
    "answer": "mangio",
}

request = {
    "level": "A1",
    "grammar_focus": "present_tense",
    "topic": "food"
}

score, breakdown = rf.score(exercise, request)

print(f"Total Score: {score}/100")
print(breakdown)
```

### Output Example

```
Total Score: 91.0/100
  JSON: 15.0/15
  Linguistic: 35.0/35
  CEFR: 17.0/20
  Fluency: 7.0/10
  Grammar: 10.0/10
  Topic: 7.0/10
Errors: Vocabulary may be too advanced for A1 (coverage: 50%)
```

### Convenience Function

```python
from src.rl.reward_function import score_exercise

# One-shot scoring (creates global instance)
score, breakdown = score_exercise(exercise, request)
```

## Individual Scorers

### 1. JSONScorer (15 points)

**Validates**:
- Required fields present (6 pts)
- Valid exercise type (3 pts)
- Type-specific fields (6 pts)

**File**: `scorers/json_scorer.py`

### 2. LinguisticScorer (35 points)

**Validates comprehensive Italian grammar**:
- Article-noun gender agreement (8 pts) - All types: definite, indefinite, partitive, elided
- Number agreement (7 pts) - Singular/plural consistency
- Adjective-noun agreement (7 pts) - Gender AND number
- Verb-subject agreement (6 pts) - Person AND number
- Preposition usage (4 pts) - Common errors (andare a/in)
- Pronoun agreement (3 pts) - Reflexive pronouns

**Features**:
- 60+ gender exceptions (la mano, il problema, aquila, etc.)
- Phonetic rules (lo studente, il gatto, l'aquila)
- Invariant adjectives (blu, rosa, viola)

**File**: `scorers/linguistic_scorer.py`

### 3. CEFRScorer (20 points)

**Validates level-appropriate complexity**:
- Sentence length (8 pts) - Different ranges per level
- Vocabulary complexity (7 pts) - Uses 16,887-word vocabulary with CEFR tags
- Grammar complexity (5 pts) - Allowed tenses per level

**Vocabulary Coverage**:
- Expects 70-90% coverage for appropriate level
- Cumulative: B1 students know A1+A2+B1 words
- Real CEFR tags (not frequency-based estimation)

**File**: `scorers/cefr_scorer.py`

### 4. FluencyScorer (10 points)

**Checks for**:
- Sentence fragments (missing verbs) - 3 pts penalty
- Excessive word repetition - 2 pts penalty
- Very short responses - 3 pts penalty
- Unnatural patterns (all caps, excessive punctuation) - 1-2 pts penalty

**File**: `scorers/fluency_scorer.py`

### 5. GrammarScorer (10 points)

**Validates requested grammar focus**:
- Tense checking (past_tense, present_tense, future_tense, imperfect_tense)
- Uses spaCy morphology + pattern indicators
- Binary: 10 points if correct, 0 if missing

**File**: `scorers/grammar_scorer.py`

### 6. TopicScorer (10 points)

**Validates topic relevance**:
- Uses sentence-transformers (paraphrase-multilingual-mpnet-base-v2)
- Computes cosine similarity between exercise and topic
- Scoring thresholds:
  - >0.7 similarity: 10 pts
  - >0.5 similarity: 7 pts
  - >0.3 similarity: 3 pts
  - ≤0.3 similarity: 0 pts

**File**: `scorers/topic_scorer.py`

## Testing

### Run Tests

```bash
# Test modular reward function
python -m src.rl.reward_function.reward_function_modular

# Test vocabulary loading
python src/rl/data/vocabulary_lists.py
```

### Test Coverage

The test suite includes:
1. ✅ A1 perfect exercise (simple present, basic vocab)
2. ✅ Article-noun gender error detection
3. ✅ Gender exception handling (la mano)
4. ✅ B1 complex sentence validation
5. ✅ Verb-subject agreement error detection

## Extending the Reward Function

### Adding a New Scorer

1. **Create scorer file** in `scorers/`
2. **Inherit from BaseScorer**:

```python
from .base import BaseScorer

class MyScorer(BaseScorer):
    def __init__(self, nlp=None):
        super().__init__(nlp)

    def score(self, exercise, request):
        # Your scoring logic
        return score, errors

    @property
    def max_score(self):
        return 15.0  # Your max score

    @property
    def name(self):
        return "my_component"
```

3. **Register in main coordinator** (`reward_function_modular.py`):

```python
self.scorers["my_component"] = MyScorer(nlp=self.nlp)
```

4. **Update RewardBreakdown** dataclass to include your component

## Performance Considerations

### Initialization Time

- **spaCy model loading**: ~1-2 seconds
- **Vocabulary loading**: ~0.5 seconds (16,887 words, cached)
- **Sentence transformer loading**: ~2-3 seconds
- **Total initialization**: ~5 seconds

### Scoring Time

- **Per exercise**: ~50-100ms
- **Bottlenecks**:
  - spaCy parsing: ~20ms
  - Sentence similarity: ~30ms (if topic specified)
  - Grammar checks: ~20ms

### Memory Usage

- **spaCy model**: ~100MB
- **Vocabulary cache**: ~2MB (16,887 words)
- **Sentence transformer**: ~400MB
- **Total**: ~500MB

## Migration from Legacy Reward Function

The original monolithic `reward_function.py` is still available but **deprecated**.

### Key Differences

| Aspect | Legacy | Modular |
|--------|--------|---------|
| Vocabulary | 200 manual words | 16,887 CEFR-tagged words |
| Architecture | Monolithic (1,400 lines) | Modular (6 scorer classes) |
| Data | Inline dictionaries | Separate data modules |
| Extensibility | Hard to extend | Easy (inherit BaseScorer) |
| Testing | One large test | Individual scorer tests |

### Migration Steps

**Old code**:
```python
from src.rl.reward_function import ExerciseRewardFunction
```

**New code** (same API!):
```python
from src.rl.reward_function import ExerciseRewardFunction
```

The API is **identical** - no code changes needed! The import just points to the new modular version.

## Data Sources

### Italian Vocabulary

- **Source**: [Language-Learning-decks](https://github.com/vbvss199/Language-Learning-decks)
- **File**: `italian/italian.json`
- **License**: MIT (repository license)
- **Format**: JSON with fields:
  - `word`: Italian word
  - `cefr_level`: A1, A2, B1, B2, C1, C2
  - `english_translation`: English meaning
  - `example_sentence_native`: Italian example
  - `example_sentence_english`: English translation
  - `pos`: Part of speech
  - `word_frequency`: Frequency rank

### Gender Exceptions

Manually curated list of 60+ Italian words with irregular gender:
- Masculine -a endings (problema, sistema, cinema)
- Feminine -o endings (mano, foto, radio)
- Animal grammatical genders (aquila, ragno, serpente)

### Article Rules

Comprehensive rules for all Italian article types:
- Definite: il, lo, la, i, gli, le
- Indefinite: un, uno, una, un'
- Partitive: del, dello, della, dei, degli, delle
- Elided: l' (with gender awareness)
- Phonetic rules: lo studente, il gatto, l'aquila

## Next Steps

1. ✅ **Modular architecture complete** - Separate scorers and data modules
2. ✅ **Comprehensive vocabulary integrated** - 16,887 CEFR-tagged words
3. ⏳ **Generate training requests dataset** - Extract from V4 or use templates
4. ⏳ **Implement GRPO training script** - Use TRL library
5. ⏳ **Train V5 model** - Target 85+ average reward score

## References

- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [Language-Learning-decks Vocabulary](https://github.com/vbvss199/Language-Learning-decks)
- [spaCy Italian Models](https://spacy.io/models/it)
- [Sentence Transformers](https://www.sbert.net/)

---

**Version**: 2.0 (Modular)
**Last Updated**: 2025-10-11
**Status**: ✅ Ready for GRPO training
