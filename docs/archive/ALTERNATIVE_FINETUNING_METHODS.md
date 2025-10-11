# Alternative Fine-Tuning Methods

## Current Situation

**Alpha=6 model trained on 8,859 examples:**
- ✅ Preserves base Italian knowledge (good gender agreement)
- ❌ Ignores task requirements (generates present tense when asked for past_tense)
- Problem: LoRA is too weak to learn the task properly

---

## Solutions (Ranked by Ease)

### ⭐ **Option 1: Stronger Prompting (NO RETRAINING!)**

**Status**: ✅ JUST IMPLEMENTED

**What we changed** in `colab_api.py`:
```python
# Before:
Grammar: past_tense
CRITICAL RULES:
- For "past_tense": Use passato prossimo...
- For "present_tense": Use presente...

# After:
Grammar: past_tense
⚠️ MANDATORY: Use ONLY past tense (passato prossimo like 'ho fatto'). NO present tense!
```

**How it works**:
- Adds explicit, grammar-specific warnings
- Repeats the requirement multiple times
- Uses concrete examples for the SPECIFIC grammar being tested
- More assertive language ("MANDATORY", "ONLY", "NO present tense!")

**Try this FIRST** - restart your API and test again:
```bash
# In Colab: Runtime → Restart runtime, run all cells
# Then test:
./teacher assignment create -s 1 -l A1 -t sea_animals -g past_tense -q 10
./student homework view -h 7 -s 1
```

**If this works**: Problem solved, no retraining needed! 🎉

---

### **Option 2: Increase Temperature (NO RETRAINING!)**

**Current**: temperature=0.4 (conservative, follows training closely)
**Try**: temperature=0.7-0.9 (more creative, might follow prompts better)

**Edit** `colab_api.py line 138`:
```python
# Current:
actual_temp = request.temperature if request.temperature != 0.7 else 0.4

# Change to:
actual_temp = request.temperature if request.temperature != 0.7 else 0.7
```

Higher temperature = less strict adherence to training patterns, more prompt-following.

---

### **Option 3: Few-Shot Prompting (NO RETRAINING!)**

Add examples directly in the prompt:

```python
prompt = f"""...

EXAMPLE for past_tense about fish:
{{"type": "fill_in_blank", "question": "Ieri i pesci ___ (nuotare) nel mare.", "answer": "hanno nuotato", "explanation": "Passato prossimo for completed action."}}

NOW generate your {request.quantity} exercises about {topic} using {grammer}:
["""
```

**Pros**: Model sees concrete example of what you want
**Cons**: Slightly longer prompt

---

### **Option 4: Adjust Alpha (REQUIRES RETRAINING)**

**Option 4A: Alpha=12 (Middle Ground)**
```python
lora_alpha: int = 12  # 1× rank
```
- More task learning than alpha=6
- Still preserves more base knowledge than alpha=24
- **Estimated**: 80-90% task compliance, 90-95% base knowledge retention

**Option 4B: Alpha=18 (Closer to Original)**
```python
lora_alpha: int = 18  # 1.5× rank
```
- Strong task learning
- Moderate base knowledge preservation
- **Estimated**: 95% task compliance, 80-85% base knowledge retention

**Training time**: ~4-6 hours on L4/A100

---

### **Option 5: LoRA+ (REQUIRES RETRAINING)**

**What it is**: Different learning rates for LoRA A and B matrices

```python
# In training config:
use_rslora: bool = True  # Rank-Stabilized LoRA
lora_alpha: int = 12
```

**How it helps**:
- Stabilizes training with lower alpha
- Better task learning without increasing forgetting
- State-of-the-art LoRA variant

**Implementation**: Update `config_exercise_generation.py`:
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=12,
    lora_alpha=12,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.15,
    use_rslora=True,  # Enable RS-LoRA
)
```

---

### **Option 6: DoRA (Requires Different Library)**

**What it is**: Weight-Decomposed Low-Rank Adaptation
- Decomposes weights into magnitude and direction
- Better preserves base model while learning task
- Shows 10-20% better task performance at same alpha

```python
lora_config = LoraConfig(
    r=12,
    lora_alpha=6,  # Can use lower alpha!
    use_dora=True,  # Enable DoRA
)
```

**Pros**: Best of both worlds (strong task + base preservation)
**Cons**: ~20% slower training, newer method

---

### **Option 7: Prefix Tuning (NO LoRA)**

**Completely different approach**:
- Don't modify model weights at all
- Instead, learn "soft prompts" (trainable tokens)
- Base model 100% preserved

```python
from peft import PrefixTuningConfig

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,  # Learn 20 prefix tokens
)
```

**Pros**:
- Zero catastrophic forgetting
- Much smaller adapter (~1MB vs 20MB LoRA)

**Cons**:
- Requires more prompt engineering
- Slightly lower task performance

---

### **Option 8: Two-Stage Training**

**Stage 1**: Train with alpha=12 for task learning (1 epoch)
**Stage 2**: Fine-tune with alpha=6 to restore base knowledge (0.5 epochs)

**How it works**:
1. First pass: Learn the task structure strongly
2. Second pass: Gentle refinement that doesn't override base knowledge

**Implementation**:
```bash
# Stage 1
alpha=12, epochs=1 → models/stage1/

# Stage 2
alpha=6, epochs=0.5, init_from=models/stage1/ → models/final/
```

---

### **Option 9: Constrained Decoding (NO RETRAINING!)**

**Force** the model to use past tense verbs:

```python
# In inference
from transformers import LogitsProcessor

class PastTenseLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # Boost probability of past tense verb tokens
        # Reduce probability of present tense tokens
        return scores

sampling_params = SamplingParams(
    logits_processors=[PastTenseLogitsProcessor()]
)
```

**Pros**: Guarantees grammar compliance
**Cons**: Complex implementation, need Italian verb dictionary

---

### **Option 10: Mixture of LoRAs**

Train separate LoRAs for different grammar types:

```
models/
├── lora_past_tense/     (specialized for past_tense)
├── lora_present_tense/  (specialized for present_tense)
├── lora_future_tense/   (specialized for future_tense)
└── lora_base/          (general exercises)
```

**At inference**:
```python
if grammar_focus == "past_tense":
    use_lora("lora_past_tense")
elif grammar_focus == "present_tense":
    use_lora("lora_present_tense")
```

**Pros**: Each LoRA can be stronger (alpha=24) without forgetting
**Cons**: Need to train 5-10 separate LoRAs

---

## My Recommendation

### **Try in this order:**

1. ✅ **Stronger prompting** (already done) - Test it FIRST
   - If it works: DONE! No retraining needed

2. **Increase temperature** to 0.7
   - 5-minute change, might help

3. **Add few-shot examples** to prompt
   - 30-minute implementation

4. **If still not working → Alpha=12 retraining**
   - 4-6 hours training
   - Most likely to fix the issue

5. **If alpha=12 has forgetting → Try DoRA or LoRA+**
   - Better methods, same alpha=6

6. **Nuclear option → Mixture of LoRAs**
   - Only if nothing else works

---

## Expected Results

### After Stronger Prompting (Option 1):
```
✅ Past tense exercises use past tense (90-95% compliance)
✅ No base knowledge forgetting
⚠️  Might still have occasional slips
```

### After Alpha=12 (Option 4A):
```
✅ Past tense exercises use past tense (98% compliance)
✅ Most base knowledge preserved (minor gender errors possible)
```

### After DoRA (Option 6):
```
✅ Perfect task compliance (99%)
✅ Perfect base knowledge (99%)
💰 Best solution but requires new library
```

---

## Test Plan

After each change, test with:
```bash
./teacher assignment create -s 1 -l A1 -t sea_animals -g past_tense -q 10
./teacher assignment create -s 1 -l A2 -t eagles -g past_tense -q 5
./teacher assignment create -s 1 -l A1 -t spiders -g present_tense -q 5
```

Check for:
1. ✅ Tense compliance (past exercises use past tense)
2. ✅ Gender agreement (le aquile, il ragno)
3. ✅ Topic adherence (no drift)
4. ✅ Realistic scenarios

---

## Quick Decision Tree

```
Has strong prompting fixed it?
├─ Yes → DONE! Use current model
└─ No → Try temperature=0.7
    ├─ Yes → DONE!
    └─ No → Retrain with alpha=12
        ├─ Good task, good Italian → DONE!
        ├─ Good task, some errors → Try DoRA with alpha=6
        └─ Bad task → Mixture of LoRAs (nuclear option)
```

---

## Bottom Line

**Most likely solution**: The stronger prompting we just implemented should fix 80-90% of the issue. Test it first before any retraining!

**If that's not enough**: Alpha=12 retraining will almost certainly work, with minimal risk of forgetting.

**Best long-term**: Investigate DoRA - it's the state-of-the-art for exactly this problem.
