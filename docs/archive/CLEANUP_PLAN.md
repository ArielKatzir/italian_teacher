# Project Cleanup Plan

**Date**: October 11, 2025

## Files to Remove (Outdated/Redundant)

### ❌ Documentation Files (Outdated V4 iterations)
- `MUST_TRAIN_NEW_MODEL.md` - Obsolete (V4 is trained)
- `ALTERNATIVE_FINETUNING_METHODS.md` - Superseded by RL_VS_ALTERNATIVES.md
- `IMPROVEMENTS_V4.md` - Historical, can archive
- `QUICK_START_V4.md` - Historical, can archive
- `COMPREHENSIVE_AUGMENTATION_V4.md` - Historical, can archive
- `RUN_V4_COMPREHENSIVE.sh` - One-time script, no longer needed

### ❌ Test Scripts (Temporary)
- `test_v4_quality.py` - Was for V4 testing, not needed for RL phase
- `src/data_generation/test_augment.py` - One-time validation script

### ❌ Dataset Directories (Old versions)
- `data/datasets/archive/` - Old datasets, not used
- `data/datasets/final/` - Superseded by v4_augmented

## Files to Keep

### ✅ Core Documentation
- `README.md` - Main project documentation
- `ROADMAP.md` - Project phases and progress
- `RL_VS_ALTERNATIVES.md` - Current decision point

### ✅ Core Code
- `src/` - All production code
- `demos/` - Colab notebooks
- `models/` - Trained models

### ✅ Configuration
- `requirements.txt` - Dependencies
- `run_api.sh` - API startup
- `teacher`, `student` - CLI helpers

### ✅ Active Data
- `data/datasets/v4_augmented/` - Current training data

## Proposed Structure After Cleanup

```
italian_teacher/
├── README.md                    # Main documentation
├── ROADMAP.md                   # Project progress
├── RL_IMPLEMENTATION.md         # New: RL guide (to be created)
├── requirements.txt
├── run_api.sh
├── teacher, student             # CLI helpers
│
├── data/
│   ├── datasets/
│   │   └── v4_augmented/       # Current data
│   └── italian_teacher.db       # SQLite database
│
├── src/
│   ├── api/                     # FastAPI backend
│   ├── cli/                     # Teacher/Student CLI
│   ├── fine_tuning/             # Training code
│   ├── data_generation/         # Data augmentation scripts
│   └── rl/                      # NEW: RL training code
│
├── models/
│   └── italian_exercise_generator_v4/    # Current model
│
└── demos/
    └── colab_inference_api.ipynb
```

## Cleanup Steps

1. ✅ Archive old documentation
2. ✅ Remove temporary scripts
3. ✅ Clean old datasets
4. ✅ Create RL directory structure
5. ✅ Update README with current status
