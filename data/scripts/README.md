# Data Processing Scripts

This directory contains scripts for dataset creation, augmentation, and processing.

## Active Scripts (Production Use)

### Dataset Processing

**`active/create_splits.py`**
- Creates stratified train/validation/test splits (80/10/10)
- Maintains CEFR level distribution across splits
- Usage: `python active/create_splits.py <input_file> <output_dir>`

**`active/fix_dataset_issues.py`**
- Validates and cleans dataset issues
- Fixes multiple-choice without options
- Removes invalid exercises
- Usage: `python active/fix_dataset_issues.py`

**`active/augment_dataset_safe.py`**
- **Safe data augmentation with incremental saves**
- Generates variations and new exercises using LLM
- Checkpoint every N examples (no data loss on crash)
- Can resume from checkpoint
- Usage: `python active/augment_dataset_safe.py --target-size 4000`

### Data Collection

**`active/parse_onlineitalianclub.py`**
- Parses HTML from onlineitalianclub.com
- Extracts fill-in-blank and multiple-choice exercises
- Handles JavaScript array formats

**`active/scrape_level.py`**
- Downloads exercises for specific CEFR level
- Polite scraping with 2-second delays
- Usage: `python active/scrape_level.py A1 https://...`

**`active/batch_extract_by_level.py`**
- Batch processes HTML files by CEFR level
- Extracts exercises using parser
- Detects exercise types automatically

## Archive (Historical Reference)

### v1.0 Dataset Creation Scripts
Located in `archive/`:

- `augment_dataset.py` - Original augmentation (unsafe, replaced by safe version)
- `merge_all_datasets.py` - Merged textbook + OnlineItalianClub data
- `downsample_and_balance.py` - Downsampled overrepresented A2 level
- `batch_extract_exercises.py` - Initial HTML extraction
- `download_exercises_by_level.py` - Downloaded exercises from web
- `scrape_all_levels.sh` - Batch scraping script
- `scrape_onlineitalianclub.py` - Original scraper

### Legacy Scripts
- `gpt4o_mini_level_specific_completion.py` - For v3 conversation datasets
- `process_extracted_units.py` - Old textbook processing
- `collection/` - Old collection utilities

## Workflow for New Dataset Version

### 1. Data Collection
```bash
# Scrape new exercises
python active/scrape_level.py C2 https://onlineitalianclub.com/...

# Extract from HTML
python active/batch_extract_by_level.py C2
```

### 2. Dataset Augmentation
```bash
# Augment with LLM (safe version)
export OPENAI_API_KEY=your-key
python active/augment_dataset_safe.py \
  --input data/datasets/final/train.jsonl \
  --output data/datasets/v2/augmented.jsonl \
  --target-size 8000 \
  --checkpoint-every 50
```

### 3. Data Cleaning
```bash
# Fix issues
python active/fix_dataset_issues.py
```

### 4. Create Splits
```bash
# Create train/val/test
python active/create_splits.py \
  data/datasets/v2/clean.jsonl \
  data/datasets/v2/
```

## Script Dependencies

### Required packages:
```bash
pip install openai requests beautifulsoup4
```

### Environment variables:
- `OPENAI_API_KEY` - For augmentation scripts using GPT-4o-mini

## Best Practices

1. **Always use safe augmentation** - `augment_dataset_safe.py` with checkpoints
2. **Validate after processing** - Run `fix_dataset_issues.py` before training
3. **Stratified splits** - Use `create_splits.py` to maintain level distribution
4. **Archive intermediate files** - Keep originals for debugging
5. **Version control** - Create new dataset versions (v2, v3, etc.)

## Notes

- All scripts use JSONL format (JSON Lines)
- Encoding: UTF-8 with fallbacks for Italian characters
- Polite scraping: 2-second delays between requests
- Checkpoint frequency: Every 10-50 examples for safety
