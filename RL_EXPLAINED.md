# Reinforcement Learning for Italian Exercise Generation - Explained Simply

**Date**: October 11, 2025

## The Core Problem RL Solves

**Supervised Learning (what we did with V4)**:
- Show model: input → output examples
- Model learns patterns in the data
- Problem: If data has errors, model learns errors
- Problem: Model can't distinguish "good" from "perfect"

**Reinforcement Learning**:
- Show model: good examples vs bad examples
- Give explicit rewards/penalties for quality
- Model learns to maximize reward
- Result: Model optimizes for the exact metrics you care about

---

## 🎮 RL Analogy: Teaching a Dog Tricks

### Supervised Learning = Show & Tell
```
You: "When I say 'sit', do this: [shows video of dog sitting]"
Dog: [Tries to mimic the video]
Problem: Dog might sit incorrectly but looks close enough to video
```

### Reinforcement Learning = Reward & Punish
```
You: "Sit!"
Dog: [Sits with back legs bent correctly] → You give treat (+10 points)
Dog: [Sits with legs straight] → No treat (0 points)
Dog: [Doesn't sit at all] → You say "no" (-5 points)

Dog learns: "Bent legs = treat, so I'll always sit with bent legs"
```

**RL for Italian exercises**:
```
Request: "Generate past tense exercise about eagles"

Model generates: "Le aquile migrano..." (present tense)
→ Reward: -5 points (wrong tense)

Model generates: "Le aquile sono migrate..." (past tense, correct gender)
→ Reward: +25 points (perfect!)

Model learns: "Past tense + correct gender = high reward"
```

---

## 🔬 Three RL Methods Explained

### Method 1: PPO (Proximal Policy Optimization) - The Standard

**How it works**:

1. **Policy Model** = Your V4 model that generates exercises
2. **Reward Model** = Separate model that scores exercise quality
3. **Training loop**:
   ```
   For each training step:
     1. Generate exercise with policy model
     2. Score it with reward model → get reward
     3. Update policy to generate higher-reward exercises
     4. But don't change too much at once (proximal = nearby)
   ```

**Visual**:
```
Request → Policy Model → Exercise
              ↓
          Reward Model → Score: 23/35
              ↓
        Update Policy (move toward higher scores)
```

**Pros**:
- ✅ Most powerful - can handle complex rewards
- ✅ Used by ChatGPT (RLHF)
- ✅ Well-tested, lots of research

**Cons**:
- ❌ Requires training TWO models (policy + reward)
- ❌ Can be unstable (might overfit to reward model)
- ❌ Slower training

**When to use**: When you have complex, hard-to-define quality criteria

---

### Method 2: DPO (Direct Preference Optimization) - The Simpler Alternative ⭐

**How it works**:

1. **No separate reward model needed!**
2. Instead, just show: "This exercise is better than that one"
3. **Training data** = pairs of (good_exercise, bad_exercise)
4. **Training loop**:
   ```
   For each pair:
     1. Model generates both exercises
     2. Calculate: P(good) vs P(bad)
     3. Update model to increase P(good) and decrease P(bad)
   ```

**Visual**:
```
Request: "A2 past_tense eagles"

Pair 1 (good):                      Pair 2 (bad):
"Le aquile sono migrate..."         "Gli aquile migrano..."
✅ Score: 35/35                      ❌ Score: 15/35
       ↓                                    ↓
Model learns to prefer left over right
```

**Training data example**:
```json
{
  "request": {"level": "A2", "grammar": "past_tense", "topic": "eagles"},
  "chosen": {
    "question": "Dove sono migrate le aquile?",
    "answer": "Le aquile sono migrate a sud per l'inverno.",
    "score": 35
  },
  "rejected": {
    "question": "Dove migrano gli aquile?",
    "answer": "Gli aquile migrano verso sud.",
    "score": 15
  }
}
```

**Pros**:
- ✅ **Simpler** - Only ONE model to train (not two)
- ✅ **More stable** - Direct optimization, less complex
- ✅ **Works with small datasets** - Just need preference pairs
- ✅ **Fast training** - 1-2 days on L4 GPU

**Cons**:
- ❌ Less flexible than PPO (can't handle super complex rewards)
- ❌ Requires good preference pairs

**When to use**: When you can clearly rank "good" vs "bad" (perfect for us!)

---

### Method 3: GRPO (Group Relative Policy Optimization) - The New Best Practice ⭐⭐

**How it works**:

1. **Generate multiple outputs** for the same request (e.g., 4-8 exercises)
2. **Score all of them** with your reward function
3. **Compare within the group** (not absolute scores)
4. **Update model** to prefer higher-scoring ones in each group

**Visual**:
```
Request: "A2 past_tense eagles"

Generate 4 exercises:
├─ Exercise A → Score: 85/100  (best in group)
├─ Exercise B → Score: 70/100
├─ Exercise C → Score: 60/100
└─ Exercise D → Score: 45/100  (worst in group)

Training signal:
"Generate more like A and B, less like C and D"
```

**Key insight**: Compare outputs to EACH OTHER, not to absolute standard

**Pros**:
- ✅ **Best of DPO + PPO** - Stable like DPO, powerful like PPO
- ✅ **No reference model needed** - Just compare within group
- ✅ **Works with ANY reward function** - Code-based or LLM-based
- ✅ **More stable than PPO** - Group comparison reduces variance
- ✅ **Simpler than DPO** - Don't need to pre-generate preference pairs
- ✅ **NEW** (2024) - State-of-the-art method used by DeepSeek, Qwen

**Cons**:
- ❌ Requires generating multiple outputs (slower inference during training)
- ❌ Newer, less documentation than DPO/PPO

**When to use**: **When you want the best results** - This is becoming the new standard

---

### Method 4: RLAIF (RL from AI Feedback) - The Automated Approach

**How it works**:

1. **Use a judge LLM** (like GPT-4) to score exercises automatically
2. **No human labeling needed!**
3. **Training loop**:
   ```
   For each training step:
     1. Generate exercise with your model
     2. Ask GPT-4: "Score this exercise (0-100)"
     3. Use GPT-4's score as reward
     4. Update model to maximize GPT-4's scores
   ```

**Visual**:
```
Request → Your Model → Exercise
              ↓
          GPT-4 Judge → "Gender: ✅, Tense: ❌, Format: ✅ → Score: 60/100"
              ↓
        Update Model (increase score)
```

**Example prompt for GPT-4 judge**:
```
Score this Italian exercise (0-100):

Request: A2 level, past_tense, eagles
Exercise: {
  "question": "Dove migrano gli aquile?",
  "answer": "Gli aquile migrano a sud."
}

Criteria:
- Gender agreement (0-25):
- Tense match (0-25):
- Italian quality (0-25):
- Topic adherence (0-25):

Output score and reasoning.
```

**Pros**:
- ✅ **No manual labeling** - GPT-4 does the scoring
- ✅ **Flexible** - Can change scoring criteria anytime
- ✅ **Good for complex criteria** - GPT-4 understands Italian grammar

**Cons**:
- ❌ **Expensive** - $0.01-0.05 per exercise scored
- ❌ **Dependent on judge quality** - If GPT-4 is wrong, model learns wrong
- ❌ **Slower** - API calls for each evaluation

**When to use**: When you have budget and need sophisticated scoring

---

## 📊 Comparison Table

| Method | Models to Train | Data Needed | Cost | Stability | Power | Year |
|--------|----------------|-------------|------|-----------|-------|------|
| **PPO** | 2 (policy + reward) | Reward examples | $10 | Medium | High | 2017 |
| **DPO** | 1 (policy only) | Preference pairs | $5 | High | Medium | 2023 |
| **GRPO** | 1 (policy only) | Reward function | $7 | **High** | **High** | **2024** |
| **RLAIF** | 1 (policy only) | None (GPT-4 judges) | $50+ | High | High | 2023 |

---

## 🎯 Which Should We Use? → **GRPO** (or DPO as backup)

### Why GRPO is Best for Us

**GRPO advantages**:
1. ✅ **Direct reward optimization** - Use our reward function directly
2. ✅ **No preference pair generation** - Train directly on-the-fly
3. ✅ **Best of both worlds** - Stable as DPO, powerful as PPO
4. ✅ **State-of-the-art** - Used by DeepSeek-R1, Qwen2.5 (2024)
5. ✅ **Group comparison** - More robust than absolute scoring

**How GRPO training works**:
```python
for batch in training_data:
    for request in batch:
        # Generate 4 exercises for same request
        exercises = [model.generate(request) for _ in range(4)]

        # Score each with reward function
        scores = [reward_function(ex, request) for ex in exercises]

        # Train: increase P(high-score), decrease P(low-score)
        # But scores are RELATIVE within this group of 4
        loss = grpo_loss(exercises, scores)
        optimizer.step()
```

**Comparison to DPO**:
- DPO: Need to pre-generate pairs, two-stage process
- GRPO: Generate and train in one go, online learning

### Why DPO is Good Backup

**1. We can easily create preference pairs**:
```python
def generate_preference_pairs():
    for request in test_requests:
        # Generate 10 candidates
        candidates = [v4_model.generate(request) for _ in range(10)]

        # Score each with our reward function
        scores = [score_exercise(c, request) for c in candidates]

        # Pick best and worst
        best = candidates[max_index(scores)]
        worst = candidates[min_index(scores)]

        yield {"chosen": best, "rejected": worst, "request": request}
```

**2. Our reward function is clear**:
- Gender agreement: spaCy check → binary (correct/wrong)
- Tense match: Pattern matching → binary (correct/wrong)
- JSON format: Schema validation → binary (valid/invalid)
- Topic adherence: Keyword presence → score 0-1

**3. We don't need complex rewards**:
- Not trying to teach creativity or style
- Just want: correct grammar + correct format + follow instructions

**4. Practical**:
- 1 week to implement
- ~$5 for training
- Uses existing V4 model

---

## 🔨 How GRPO Works Step-by-Step (Recommended)

### What Makes GRPO Special?

**Key Innovation**: Instead of comparing to a fixed reference (like DPO) or using a separate reward model (like PPO), GRPO compares multiple outputs **from the same model** for the **same request**.

**Intuition**:
```
Old approach (DPO): "This exercise is good, that one is bad" (absolute judgment)
GRPO: "Among these 4 exercises I just generated, rank them" (relative judgment)
```

Why is relative better?
- ✅ More robust (comparing apples to apples)
- ✅ Reduces reward hacking (model can't exploit fixed "good" examples)
- ✅ Adapts as model improves (always comparing current outputs)

### Step 1: Create Reward Function (Same as DPO)

```python
def reward_function(exercise, request):
    """Score exercise quality (0-100)."""
    score = 0

    # Gender validation (25 points)
    if all_articles_match_noun_gender(exercise):
        score += 25

    # Tense validation (25 points)
    if exercise_tense_matches_grammar_focus(exercise, request.grammar_focus):
        score += 25

    # JSON structure (25 points)
    if valid_json_schema(exercise):
        score += 25

    # Topic adherence (25 points)
    if topic_similarity(exercise.question, request.topic) > 0.7:
        score += 25

    return score
```

### Step 2: GRPO Training Loop (Online, No Pre-generation)

```python
from trl import GRPOTrainer
from transformers import AutoModelForCausalLM

# Load V4 model
model = AutoModelForCausalLM.from_pretrained(
    "models/italian_exercise_generator_v4_merged"
)

# GRPO Configuration
grpo_config = GRPOConfig(
    num_sample_generations=4,  # Generate 4 exercises per request
    learning_rate=1e-6,
    kl_coef=0.1,  # KL penalty to prevent drift
    max_length=2048,
)

# Create trainer
trainer = GRPOTrainer(
    model=model,
    config=grpo_config,
    reward_function=reward_function,  # Our custom reward function!
    train_dataset=training_requests,  # Just requests, not full exercises
)

# Train (generates exercises on-the-fly during training)
trainer.train()
```

### Step 3: What Happens During Training

**For each training step**:

```python
# 1. Sample a request from training data
request = {
    "level": "A2",
    "grammar": "past_tense",
    "topic": "eagles"
}

# 2. Generate 4 exercises with current model
exercises = []
for _ in range(4):
    ex = model.generate(request)
    exercises.append(ex)

# Example outputs:
# Exercise 0: "Le aquile sono migrate a sud." → Score: 85
# Exercise 1: "Gli aquile migrano a sud." → Score: 45
# Exercise 2: "L'aquila è migrata verso regioni calde." → Score: 90
# Exercise 3: "Le aquile migrano in inverno." → Score: 60

# 3. Score each exercise
scores = [reward_function(ex, request) for ex in exercises]
# scores = [85, 45, 90, 60]

# 4. Compute advantages (how much better than average in THIS group)
mean_score = np.mean(scores)  # 70
advantages = [s - mean_score for s in scores]
# advantages = [+15, -25, +20, -10]

# 5. Update model
# Increase probability of Exercise 2 (best: +20)
# Increase probability of Exercise 0 (good: +15)
# Decrease probability of Exercise 3 (below avg: -10)
# Decrease probability of Exercise 1 (worst: -25)

loss = grpo_loss(exercises, advantages)
optimizer.step()
```

**Key difference from DPO**:
- DPO: Needs pre-generated (chosen, rejected) pairs
- GRPO: Generates and learns in one pass, adapts to current model

### Step 4: Training Data Format (Simpler!)

GRPO only needs **requests**, not full exercise examples:

```jsonl
{"level": "A2", "grammar": "past_tense", "topic": "eagles"}
{"level": "B1", "grammar": "present_tense", "topic": "cooking"}
{"level": "A1", "grammar": "definite_articles", "topic": "family"}
...
```

That's it! No need to generate thousands of exercise pairs ahead of time.

### Step 5: Advantages Over DPO

**DPO workflow**:
```
1. Generate 10,000 exercises offline (1-2 days)
2. Score all of them
3. Create preference pairs (best vs worst)
4. Train model on pairs (1-2 days)
Total: 3-4 days
```

**GRPO workflow**:
```
1. Train model (generates 4 per request, scores, learns immediately)
Total: 2-3 days

Everything happens during training!
```

**Benefits**:
- ✅ Faster pipeline (no offline generation)
- ✅ Model adapts to its own improving quality
- ✅ More data-efficient (sees many variations during training)

---

## 🔨 How DPO Works Step-by-Step (Backup Option)

### Step 1: Create Reward Function

```python
def reward_function(exercise, request):
    """Score exercise quality (0-100)."""
    score = 0

    # Gender validation (25 points)
    if all_articles_match_noun_gender(exercise):
        score += 25

    # Tense validation (25 points)
    if exercise_tense_matches_grammar_focus(exercise, request.grammar_focus):
        score += 25

    # JSON structure (25 points)
    if valid_json_schema(exercise):
        score += 25

    # Topic adherence (25 points)
    if topic_similarity(exercise.question, request.topic) > 0.7:
        score += 25

    return score
```

### Step 2: Generate Preference Dataset

```python
preference_dataset = []

for request in training_requests:  # 1000 requests
    # Generate 10 variations
    candidates = []
    for _ in range(10):
        exercise = v4_model.generate(request)
        score = reward_function(exercise, request)
        candidates.append((exercise, score))

    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Take best and worst
    chosen = candidates[0][0]  # Highest score
    rejected = candidates[-1][0]  # Lowest score

    # Only include if there's clear difference
    if candidates[0][1] - candidates[-1][1] > 20:
        preference_dataset.append({
            "request": request,
            "chosen": chosen,
            "rejected": rejected,
        })

# Result: ~800-900 high-quality preference pairs
```

### Step 3: Train with DPO

```python
from trl import DPOTrainer
from transformers import AutoModelForCausalLM

# Load V4 model
model = AutoModelForCausalLM.from_pretrained(
    "models/italian_exercise_generator_v4_merged"
)
ref_model = AutoModelForCausalLM.from_pretrained(
    "models/italian_exercise_generator_v4_merged"
)  # Frozen reference

# Setup DPO trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=preference_dataset,
    beta=0.1,  # KL penalty (keep close to original)
    max_length=2048,
    learning_rate=1e-6,
)

# Train
trainer.train()

# Save
model.save_pretrained("models/italian_exercise_generator_v5_dpo")
```

### Step 4: Evaluate

```python
# Test on problematic cases
test_cases = [
    {"level": "A2", "grammar": "past_tense", "topic": "eagles"},
    {"level": "A2", "grammar": "past_tense", "topic": "spiders"},
]

for test in test_cases:
    exercise = v5_dpo_model.generate(test)
    score = reward_function(exercise, test)
    print(f"Score: {score}/100")
    print(f"Exercise: {exercise}")
```

---

## 🎓 Key Concepts Explained

### 1. What is a "Policy"?
- **Policy** = The model's behavior (how it generates exercises)
- "Update the policy" = Change model weights so it behaves better
- Goal: Policy that generates high-reward exercises

### 2. What is "KL Divergence"?
- **KL** = Measure of how different two probability distributions are
- In DPO: How much did the model change from original?
- **KL penalty** = Punishment for changing too much
- Why? Prevent model from forgetting Italian while learning task

### 3. What is a "Reward Model"?
- Separate model that predicts: "Is this exercise good?"
- Trained on human-labeled examples
- Used in PPO, not needed in DPO

### 4. What are "Preference Pairs"?
- Two exercises, one better than the other
- Example: (good_exercise, bad_exercise)
- DPO learns to prefer the good one

---

## 🚀 Implementation Timeline

### Week 1: Reward Function & Data Generation
- **Day 1-2**: Build comprehensive reward function
  - spaCy gender validation
  - Tense pattern matching
  - JSON schema validation
  - Topic similarity scoring
- **Day 3-5**: Generate preference dataset
  - 1000 requests × 10 candidates = 10,000 generations
  - Score and rank
  - Create ~800-900 preference pairs
- **Day 6-7**: Validate data quality
  - Manual spot-checks
  - Ensure score differences are meaningful

### Week 2: DPO Training
- **Day 1**: Setup TRL DPOTrainer
- **Day 2-3**: Train on L4 GPU (36-48 hours)
- **Day 4-5**: Evaluate and iterate
- **Day 6-7**: Deploy V5 model

---

## 💡 Why This Will Work

**Current problem with V4**:
- Model learned patterns from data
- Data had some errors → model learned errors
- No explicit feedback on quality

**With DPO**:
- Model sees: "This is good, that is bad"
- Reward function enforces correct Italian grammar rules
- Model optimizes directly for correctness

**Expected results**:
- V4: 85% gender accuracy → V5: 95%+
- V4: 75% tense consistency → V5: 95%+
- V4: 90% JSON validity → V5: 99%+

---

## 📚 Further Reading

- [DPO Paper](https://arxiv.org/abs/2305.18290): Direct Preference Optimization
- [TRL Library](https://github.com/huggingface/trl): HuggingFace RL tools
- [RLHF Blog](https://huggingface.co/blog/rlhf): Reinforcement Learning from Human Feedback

---

## Next Steps

1. **Agree on DPO approach** ✓
2. **Build reward function** (I can help design this)
3. **Generate preference dataset** (automated script)
4. **Train with DPO** (2 days on Colab)
5. **Evaluate improvements** (compare V4 vs V5)

**Ready to start building the reward function?**
