#!/bin/bash

# VoiceFilter Dataset Subset Creation Script for Vertex AI
# Run this on a Vertex AI Notebook or Compute Engine instance

echo "🚀 VoiceFilter Dataset Subset Creator"
echo "📅 $(date)"

# Install required packages
echo "📦 Installing required packages..."
pip install google-cloud-storage

# Set your GCS bucket names
SOURCE_BUCKET="voicefilter-data"  # Your existing bucket
TARGET_BUCKET="voicefilter-data"  # Same bucket, different prefix
SOURCE_TRAIN_PREFIX="vf_data_production/train"
SOURCE_TEST_PREFIX="vf_data_production/test"
TARGET_PREFIX="vf_data_subsets"
SUBSET_SIZE_GB=15

echo "📂 Source: gs://${SOURCE_BUCKET}/${SOURCE_TRAIN_PREFIX}"
echo "📂 Target: gs://${TARGET_BUCKET}/${TARGET_PREFIX}"

# First, run a dry run to analyze the dataset
echo "🔍 Running analysis (dry run)..."
python create_gcs_subsets.py \
    --source-bucket "$SOURCE_BUCKET" \
    --source-train-prefix "$SOURCE_TRAIN_PREFIX" \
    --source-test-prefix "$SOURCE_TEST_PREFIX" \
    --target-bucket "$TARGET_BUCKET" \
    --target-prefix "$TARGET_PREFIX" \
    --subset-size-gb $SUBSET_SIZE_GB \
    --dry-run

echo ""
echo "❓ Do you want to proceed with creating the subsets? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "🔄 Creating subsets..."
    python create_gcs_subsets.py \
        --source-bucket "$SOURCE_BUCKET" \
        --source-train-prefix "$SOURCE_TRAIN_PREFIX" \
        --source-test-prefix "$SOURCE_TEST_PREFIX" \
        --target-bucket "$TARGET_BUCKET" \
        --target-prefix "$TARGET_PREFIX" \
        --subset-size-gb $SUBSET_SIZE_GB
    
    echo "✅ Subset creation completed!"
    echo "📊 Your subsets are now available at:"
    echo "   gs://${TARGET_BUCKET}/${TARGET_PREFIX}/subset_01/"
    echo "   gs://${TARGET_BUCKET}/${TARGET_PREFIX}/subset_02/"
    echo "   gs://${TARGET_BUCKET}/${TARGET_PREFIX}/subset_03/"
    echo "   ... and so on"
    
else
    echo "❌ Subset creation cancelled."
fi

echo "🏁 Script completed at $(date)" 