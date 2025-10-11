# Comprehensive V4 Data Augmentation

**Goal**: Massive vocabulary & grammar coverage to prevent catastrophic forgetting

## What's Being Generated

### ~5,700 new training examples total:

**1. Topic-based examples (2,000)**
- 20 diverse categories × 100 samples each
- Focus: **Vocabulary breadth**

**2. Grammar-based examples (3,700)**
- 74 grammar focuses × 50 samples each
- Focus: **Grammatical coverage**

---

## 20 Topic Categories

All aspects of Italian vocabulary:

1. **animals_wildlife** - animals, wildlife, insects, birds, marine life, pets
2. **nature_environment** - trees, plants, flowers, landscapes, weather, seasons
3. **food_cooking** - ingredients, dishes, cooking methods, restaurants, recipes
4. **professions_work** - jobs, workplaces, professional activities, careers
5. **sports_fitness** - sports, exercises, competitions, athletes, equipment
6. **travel_geography** - countries, cities, landmarks, transportation, tourism
7. **home_furniture** - rooms, furniture, appliances, household items
8. **clothing_fashion** - clothes, accessories, fabrics, styles, shopping
9. **technology_digital** - computers, internet, devices, apps, software
10. **arts_culture** - painting, music, theater, cinema, literature, museums
11. **education_learning** - schools, subjects, studying, exams, teachers
12. **health_medicine** - body parts, illnesses, treatments, doctors, hospitals
13. **family_relationships** - family, relationships, emotions, life events
14. **hobbies_leisure** - activities, games, entertainment, collections, crafts
15. **science_research** - physics, chemistry, biology, experiments, scientists
16. **history_society** - historical events, civilizations, politics, traditions
17. **economy_finance** - money, banking, trade, investments, markets
18. **law_justice** - legal system, rights, courts, laws, regulations
19. **vehicles_transport** - cars, trains, planes, boats, transportation systems
20. **abstract_concepts** - time, space, emotions, ideas, philosophy, logic

---

## 74 Grammar Focuses

Complete Italian grammar coverage:

### 🟦 Morfologia (13 focuses)
- noun_gender, noun_number, irregular_nouns
- definite_articles, indefinite_articles, partitive_articles
- qualitative_adjectives, demonstrative_adjectives, possessive_adjectives, adjective_degrees
- adverbs, adverb_comparison
- subject_pronouns, object_pronouns, reflexive_pronouns, possessive_pronouns, demonstrative_pronouns, relative_pronouns, interrogative_pronouns, indefinite_pronouns

### 🟥 Verbi - Indicativo (8 focuses)
- present_tense, imperfect_tense, past_tense, pluperfect
- simple_past, remote_pluperfect
- future_tense, future_perfect

### 🟥 Verbi - Congiuntivo (4 focuses)
- subjunctive_present, subjunctive_imperfect
- subjunctive_past, subjunctive_pluperfect

### 🟥 Verbi - Other Moods (7 focuses)
- conditional_present, conditional_past
- imperative
- infinitive, gerund, participle
- sequence_of_tenses

### 🟥 Verbi - Special Structures (5 focuses)
- passive_voice, reflexive_verbs, pronominal_verbs
- modal_verbs, impersonal_verbs

### 🟩 Sintassi (8 focuses)
- simple_sentence, complex_sentence, complements
- si_impersonal, si_passive
- subject_verb_agreement, noun_adjective_agreement, word_order

### 🟨 Proposizioni (9 focuses)
- temporal_clauses, causal_clauses, final_clauses
- consecutive_clauses, concessive_clauses, conditional_clauses
- relative_clauses, declarative_clauses, indirect_questions

### 🟪 Strutture Complesse (5 focuses)
- direct_indirect_speech
- hypothetical_type1, hypothetical_type2, hypothetical_type3
- subjunctive_in_subordinates

### 🟫 Ortografia (4 focuses)
- accents, apostrophe_elision, double_consonants, h_usage

### ⚫ Lessico (4 focuses)
- synonyms_antonyms, homophones, idiomatic_expressions, proverbs

---

## Why This Approach

### Problem with Previous Approach
- Only 4 topic categories (animals, nature, food, professions)
- Only 7 grammar focuses emphasized
- Total: 500 examples - **not enough vocabulary diversity**
- Model still encounters out-of-vocabulary words → gender errors

### Comprehensive Approach Benefits
- **20 topic categories** = massive vocabulary coverage
- **74 grammar focuses** = every aspect of Italian grammar
- **5,700 examples** = 11× more than previous approach
- **No "critical words"** - everything is equally important

### Philosophy
> "There are no critical words. Everything is critical and nothing is critical at the same time."

The goal is **comprehensive coverage**, not targeting specific words like "ragno" or "lombrico". With 5,700 diverse examples, the model will see:
- Thousands of different nouns with correct gender
- All verb tenses in natural contexts
- All grammatical structures in use
- Diverse vocabulary across all domains

---

## Cost Estimation

**GPT-4o-mini pricing**:
- Input: $0.15 / 1M tokens
- Output: $0.60 / 1M tokens

**For 5,700 examples**:
- Input: ~5,700 × 300 tokens = 1.7M tokens = $0.26
- Output: ~5,700 × 800 tokens = 4.6M tokens = $2.76
- **Total: ~$3.00**

**Time estimate**: ~50 minutes (with 0.1s delay between requests)

---

## Usage

### Run Generation

```bash
# Set API key
export OPENAI_API_KEY='your-key-here'

# Run comprehensive augmentation (~50 minutes, $3)
cd "/Users/arielkatzir/Library/CloudStorage/GoogleDrive-ari.katzir@gmail.com/My Drive/Colab Notebooks/italian_teacher"
~/.venvs/py312/bin/python src/data_generation/augment_comprehensive.py
```

### Expected Output

```
🚀 Comprehensive Italian Exercise Augmentation
   Topic categories: 20 × 100 = 2,000
   Grammar focuses: 74 × 50 = 3,700
   Total target: 5,700
   Output: data/datasets/v4_augmented/train_augmentation_comprehensive.jsonl

📝 Generating prompt specifications...
✅ Generated 5700 total prompts

🤖 Generating exercises with GPT-4o-mini...
   (This will take ~47 minutes)

   [1/5700] A2 - animals wildlife - present_tense ✅
   [2/5700] B1 - food cooking - subjunctive_present ✅
   [3/5700] A1 - nature environment - past_tense ✅
   ...
   [100/5700] ...
   💾 Saved 100 examples (checkpoint)
   ...

✅ Generated 5465 examples
❌ Failed 235 examples (4.1%)

📊 Final Statistics:
   Total examples: 5465
   Grammar coverage: 74 unique
   Topic coverage: 20 unique
```

### Merge with Original Dataset

```bash
# Merge training data
cat data/datasets/final/train.jsonl \
    data/datasets/v4_augmented/train_augmentation_comprehensive.jsonl \
    > data/datasets/v4_augmented/train.jsonl

# Verify
echo "Original: $(wc -l < data/datasets/final/train.jsonl)"
echo "Augmented: $(wc -l < data/datasets/v4_augmented/train_augmentation_comprehensive.jsonl)"
echo "Total: $(wc -l < data/datasets/v4_augmented/train.jsonl)"
```

**Expected**:
```
Original: 3186
Augmented: 5465
Total: 8651
```

---

## What This Fixes

### Before (3,186 examples, limited vocabulary)
- ❌ Gender errors on uncommon words ("gli aquile", "la ragno")
- ❌ Limited vocabulary (~10,345 unique words)
- ❌ Grammar gaps (only 124 past_tense examples)
- ❌ Topic limitations (mostly common topics)

### After (8,651 examples, comprehensive coverage)
- ✅ Massive vocabulary (~25,000+ unique words)
- ✅ Every grammar structure covered (74 focuses × 50)
- ✅ Diverse topics (20 categories × 100)
- ✅ No vocabulary gaps - model sees Italian in all contexts

---

## Combined with V4 LoRA Config

**This data augmentation works with**:
- ✅ Alpha = 6 (weaker, preserves base knowledge)
- ✅ Target modules = 2 (less interference)
- ✅ Enhanced prompts (tense enforcement)
- ✅ spaCy validation (gender + tense checking)

**Result**: Model learns task structure from diverse data while preserving base Italian knowledge

---

## Success Metrics

After V4 training with comprehensive dataset:

1. **Vocabulary coverage**: Handle any Italian word correctly
2. **Grammar accuracy**: >95% on all 74 grammar focuses
3. **Gender agreement**: 100% correct (even rare words)
4. **Tense consistency**: 100% matching grammar_focus
5. **Topic adherence**: No drift, stays on topic

**Philosophy**: With 5,700 diverse examples covering all Italian grammar and vocabulary domains, the model will have enough exposure to preserve its base knowledge while learning the exercise generation task.

---

## Files

- [augment_comprehensive.py](src/data_generation/augment_comprehensive.py) - Generation script
- Output: `data/datasets/v4_augmented/train_augmentation_comprehensive.jsonl`
- Final merged: `data/datasets/v4_augmented/train.jsonl` (8,651 examples)

---

## Alternative: Smaller Scale

If you want to test first with a smaller dataset:

```python
# In augment_comprehensive.py, change:
SAMPLES_PER_TOPIC = 50  # Instead of 100
SAMPLES_PER_GRAMMAR = 25  # Instead of 50

# Result: ~2,850 examples (~$1.50, ~25 minutes)
```

Then scale up if results are good.
