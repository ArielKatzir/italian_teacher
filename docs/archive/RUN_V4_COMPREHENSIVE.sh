#!/bin/bash
#
# V4 Comprehensive Training Pipeline
# Complete workflow from data generation to trained model
#

set -e  # Exit on error

echo "🚀 V4 Comprehensive Training Pipeline"
echo ""

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "   Set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Paths
PROJECT_ROOT="/Users/arielkatzir/Library/CloudStorage/GoogleDrive-ari.katzir@gmail.com/My Drive/Colab Notebooks/italian_teacher"
PYTHON="$HOME/.venvs/py312/bin/python"

cd "$PROJECT_ROOT"

echo "📍 Working directory: $PWD"
echo ""

# Step 1: Generate comprehensive augmented data
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Generate Augmented Data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will:"
echo "  - Generate ~5,700 new examples"
echo "  - Cover 20 topic categories × 100 samples"
echo "  - Cover 74 grammar focuses × 50 samples"
echo "  - Cost: ~\$3.00 (GPT-4o-mini)"
echo "  - Time: ~50 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

$PYTHON src/data_generation/augment_comprehensive.py

echo ""
echo "✅ Step 1 complete"
echo ""

# Step 2: Verify generated data
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Verify Generated Data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

AUGMENTED_FILE="data/datasets/v4_augmented/train_augmentation_comprehensive.jsonl"

if [ ! -f "$AUGMENTED_FILE" ]; then
    echo "❌ Error: $AUGMENTED_FILE not found"
    exit 1
fi

AUGMENTED_COUNT=$(wc -l < "$AUGMENTED_FILE")
echo "Generated examples: $AUGMENTED_COUNT"
echo ""

# Sample one example
echo "Sample example:"
head -1 "$AUGMENTED_FILE" | jq '.metadata'
echo ""
echo "✅ Step 2 complete"
echo ""

# Step 3: Merge datasets
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Merge with Original Dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ORIGINAL_TRAIN="data/datasets/final/train.jsonl"
MERGED_TRAIN="data/datasets/v4_augmented/train.jsonl"

# Create merged training set
cat "$ORIGINAL_TRAIN" "$AUGMENTED_FILE" > "$MERGED_TRAIN"

# Copy validation and test
cp data/datasets/final/validation.jsonl data/datasets/v4_augmented/
cp data/datasets/final/test.jsonl data/datasets/v4_augmented/

# Show counts
ORIGINAL_COUNT=$(wc -l < "$ORIGINAL_TRAIN")
MERGED_COUNT=$(wc -l < "$MERGED_TRAIN")

echo "Dataset sizes:"
echo "  Original:  $ORIGINAL_COUNT examples"
echo "  Augmented: $AUGMENTED_COUNT examples"
echo "  Merged:    $MERGED_COUNT examples"
echo ""
echo "✅ Step 3 complete"
echo ""

# Step 4: Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Data Preparation Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo ""
echo "1. Upload to Colab (already synced via Google Drive)"
echo ""
echo "2. Open Colab training notebook"
echo ""
echo "3. Update config paths:"
echo "   train_file = 'data/datasets/v4_augmented/train.jsonl'"
echo "   validation_file = 'data/datasets/v4_augmented/validation.jsonl'"
echo ""
echo "4. Verify LoRA config:"
echo "   alpha = 6 ✅"
echo "   target_modules = ['q_proj', 'v_proj'] ✅"
echo ""
echo "5. Run training (~2-3 hours on L4/A100)"
echo ""
echo "6. Test with:"
echo "   ./teacher homework create --level A2 --grammar past_tense --topic eagles"
echo ""
echo "📊 Expected improvements:"
echo "   ✅ Massive vocabulary coverage (~25,000+ words)"
echo "   ✅ All 74 grammar structures covered"
echo "   ✅ No gender errors (even rare words)"
echo "   ✅ 100% tense consistency"
echo "   ✅ No topic drift"
echo ""
