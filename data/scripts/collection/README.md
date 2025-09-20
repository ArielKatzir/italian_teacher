# Italian Teacher Data Collection Scripts

## 🚀 Active Scripts (Use These)

These scripts collect authentic Italian teaching data without template artifacts:

### Essential Collection Pipeline

```bash
# Activate environment
source ~/.venvs/py312/bin/activate
cd italian_teacher

# 1. Process CELI corpus (authentic learner language)
python data/scripts/collection/process_celi_corpus.py

# 2. Collect CIMA tutoring data (real teacher responses)
python data/scripts/collection/collect_cima_tutoring.py

# 3. Collect Italian conversations (natural dialogues)
python data/scripts/collection/collect_italian_conversations.py

# 4. Combine all sources into final training dataset
python data/scripts/create_final_authentic_dataset.py
```

### Script Details

| Script | Purpose | Output | Authentic? |
|--------|---------|--------|------------|
| `process_celi_corpus.py` | Process CELI corpus KWIC files | 6,329 conversations | ✅ Real learner language |
| `collect_cima_tutoring.py` | Download CIMA tutoring dataset | 5,446 conversations | ✅ Real teacher responses |
| `collect_italian_conversations.py` | Download HF Italian conversations | 3,000 conversations | ✅ Natural dialogues |

## 📁 File Structure After Collection

```
data/
├── raw/
│   ├── celi_corpus/                 # CELI KWIC files (B1-C2)
│   ├── cima_tutoring/               # CIMA tutoring conversations
│   ├── italian_conversations/       # HF conversations dataset
│   └── textbook_content/            # A1/A2 sample content
├── processed/
│   ├── celi_authentic/              # Processed CELI conversations
│   ├── celi_training_ready/         # CELI in training format
│   └── complete_a1_c2/              # 🎯 FINAL TRAINING DATASET
└── scripts/
    ├── collection/                  # This directory
    ├── create_complete_dataset.py   # Final dataset combiner
    └── convert_celi_to_training.py  # CELI format converter
```

## 🎯 Expected Results

After running the complete pipeline:

- **Total**: 15,275 authentic conversations
- **CEFR Coverage**: A1-C2 complete span (69.4% B1/B2 focus)
- **Quality**: 96.7% authentic data, 3.3% essential A1 synthetic
- **Training Ready**: HuggingFace chat format with metadata

## ❌ Deprecated Scripts

Moved to `deprecated/` folder - these scripts generated synthetic/template data that caused v1 training issues:

- `collect_babbel_podcasts.py` - Replaced with authentic tutoring
- `collect_tatoeba.py` - Replaced with natural conversations
- `collect_opus.py` - Not needed for authentic approach
- `collect_italiano_bello.py` - Synthetic content
- `collect_onlineitalianclub.py` - Synthetic content
- `collect_spoken_italian.py` - Synthetic content
- `collect_textbook_content.py` - Only 6 examples, not worth including
- `setup_collection.py` - Old pipeline setup

## 🚀 Next Step: Training

Once collection is complete, the training configuration automatically uses the authentic dataset:

```python
# src/fine_tuning/config.py already updated to use:
train_file: "data/processed/complete_a1_c2/train.jsonl"
validation_file: "data/processed/complete_a1_c2/validation.jsonl"
test_file: "data/processed/complete_a1_c2/test.jsonl"
```

**Ready for Marco v2 LoRA training with authentic data!** 🇮🇹