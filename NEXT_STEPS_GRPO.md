# Next Steps: GRPO Implementation

**Date**: October 11, 2025
**Decision**: Use GRPO for V5 model training

---

## ✅ Cleanup Complete

**Removed**:
- Outdated V4 documentation files → moved to `docs/archive/`
- Temporary test scripts
- Old cleanup plans

**Current structure**:
```
italian_teacher/
├── README.md
├── ROADMAP.md
├── RL_VS_ALTERNATIVES.md    # Why RL?
├── RL_EXPLAINED.md           # How GRPO works
├── src/rl/                   # NEW: RL implementation (empty, ready)
└── models/italian_exercise_generator_v4/  # Base for GRPO training
```

---

## 🎯 Why GRPO?

| Feature | DPO | GRPO |
|---------|-----|------|
| Workflow | Two-stage (generate → train) | One-stage (online) |
| Data needed | Preference pairs | Just requests |
| Training time | 3-4 days | 2-3 days |
| Adaptation | Fixed pairs | Adapts as model improves |
| Stability | High | High |
| Power | Medium | High |

**GRPO = Best of DPO + PPO without their downsides**

---

## 📋 Implementation Plan

### Phase 1: Reward Function (Week 1)

**Goal**: Build comprehensive scoring system for Italian exercises

**Tasks**:
1. **Gender validation** (25 points)
   - Use spaCy Italian NLP
   - Check article-noun agreement
   - Validate adjective agreement

2. **Tense validation** (25 points)
   - Pattern matching for verb forms
   - Ensure match with `grammar_focus`
   - Handle compound tenses (passato prossimo, etc.)

3. **JSON structure** (25 points)
   - Schema validation
   - Required fields present
   - Valid exercise types

4. **Topic adherence** (25 points)
   - Semantic similarity (sentence embeddings)
   - Keyword matching
   - Relevance scoring

**Output**: `src/rl/reward_function.py`

---

### Phase 2: Training Requests Dataset (Week 1)

**Goal**: Create diverse training requests for GRPO

**Format** (simple!):
```jsonl
{"level": "A2", "grammar": "past_tense", "topic": "eagles", "num_exercises": 3}
{"level": "B1", "grammar": "present_tense", "topic": "cooking", "num_exercises": 3}
{"level": "A1", "grammar": "definite_articles", "topic": "family", "num_exercises": 3}
```

**Generation strategy**:
```python
# Extract requests from existing V4 training data
# Plus add diverse combinations

levels = ["A1", "A2", "B1", "B2"]
grammars = [all 74 grammar focuses from COMPREHENSIVE_AUGMENTATION_V4]
topics = [all 20 topic categories from COMPREHENSIVE_AUGMENTATION_V4]

# Generate ~2000-3000 diverse request combinations
```

**Output**: `data/rl/training_requests.jsonl`

---

### Phase 3: GRPO Training (Week 2)

**Goal**: Train V5 model with GRPO

**Setup**:
```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM
from src.rl.reward_function import reward_function

# Load V4 as starting point
model = AutoModelForCausalLM.from_pretrained(
    "models/italian_exercise_generator_v4_merged",
    torch_dtype="float16",
    device_map="auto"
)

# GRPO Config
config = GRPOConfig(
    num_sample_generations=4,       # Generate 4 exercises per request
    learning_rate=1e-6,
    kl_coef=0.1,                   # KL penalty (stay close to V4)
    max_length=2048,
    per_device_train_batch_size=1,  # GPU memory constraint
    gradient_accumulation_steps=8,
    num_train_epochs=1,
)

# Train
trainer = GRPOTrainer(
    model=model,
    config=config,
    reward_function=reward_function,
    train_dataset=training_requests,
)

trainer.train()
model.save_pretrained("models/italian_exercise_generator_v5_grpo")
```

**Training details**:
- GPU: L4 or A100 on Colab
- Time: 2-3 days
- Cost: ~$7 (Colab Pro)

**Output**: `models/italian_exercise_generator_v5_grpo/`

---

### Phase 4: Evaluation (Week 2-3)

**Goal**: Measure improvements over V4

**Test cases**:
```python
test_cases = [
    # Known problematic cases from V4
    {"level": "A2", "grammar": "past_tense", "topic": "eagles"},  # Gender errors
    {"level": "A2", "grammar": "past_tense", "topic": "spiders"},  # Rare words
    {"level": "B1", "grammar": "subjunctive", "topic": "cooking"},  # Complex grammar
]
```

**Metrics**:
| Metric | V4 Baseline | V5 Target |
|--------|-------------|-----------|
| Gender accuracy | 85% | **95%+** |
| Tense consistency | 75% | **95%+** |
| JSON validity | 90% | **99%+** |
| Topic adherence | 80% | **90%+** |
| Overall reward | 60/100 | **85/100** |

**Output**: `docs/V5_EVALUATION.md`

---

## 🛠️ Libraries & Dependencies

**Add to requirements.txt**:
```txt
# RL Training
trl>=0.8.0              # Transformers Reinforcement Learning
peft>=0.10.0            # LoRA support
accelerate>=0.27.0      # Distributed training

# Reward function
spacy>=3.7.0            # Italian NLP
it-core-news-sm         # Italian language model
sentence-transformers   # Semantic similarity
```

**Install**:
```bash
pip install trl peft accelerate spacy sentence-transformers
python -m spacy download it_core_news_sm
```

---

## 📊 Expected Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1**: Reward function | 3-5 days | `src/rl/reward_function.py` |
| **Phase 2**: Training requests | 1-2 days | `data/rl/training_requests.jsonl` |
| **Phase 3**: GRPO training | 2-3 days | `models/italian_exercise_generator_v5_grpo/` |
| **Phase 4**: Evaluation | 2-3 days | `docs/V5_EVALUATION.md` |
| **Total** | **~2 weeks** | Production-ready V5 model |

---

## 💰 Cost Estimate

| Item | Cost |
|------|------|
| Colab Pro subscription | $10/month (or use free L4 with limits) |
| GRPO training (2-3 days) | ~$0-7 |
| OpenAI for eval (optional) | ~$1 |
| **Total** | **$11-18** |

---

## 🚀 Ready to Start?

**Next immediate action**:
1. Build reward function (`src/rl/reward_function.py`)
2. Test it on V4 outputs to validate scoring
3. Generate training requests dataset

**Would you like me to**:
- [ ] Create the reward function skeleton
- [ ] Generate training requests dataset
- [ ] Set up GRPO training script
- [ ] All of the above

Let me know and I'll start implementing!
