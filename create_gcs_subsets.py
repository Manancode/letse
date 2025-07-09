#!/usr/bin/env python3
"""
VoiceFilter Dataset Subset Creator for GCS
Organizes training data into manageable subsets while keeping complete sample groups.

Each sample needs ALL these files:
- 000000-dvec.txt
- 000000-mixed-features.pt  
- 000000-mixed.pt
- 000000-mixed.wav
- 000000-noise-type.pt
- 000000-target-features.pt
- 000000-target.pt
- 000000-target.wav

This script groups by sample ID and creates subsets of complete samples.
"""

import os
import argparse
from collections import defaultdict
from google.cloud import storage
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset_structure(client, bucket_name, train_prefix):
    """Analyze the dataset to find complete samples."""
    logger.info("🔍 Analyzing dataset structure...")
    
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=train_prefix)
    
    # Group files by sample ID
    samples = defaultdict(list)
    total_files = 0
    
    for blob in blobs:
        if blob.name.endswith('/'):  # Skip directories
            continue
            
        filename = os.path.basename(blob.name)
        if filename.startswith('0') and '-' in filename:  # Valid sample file
            sample_id = filename.split('-')[0]  # Extract "000000" from "000000-mixed.wav"
            samples[sample_id].append(blob.name)
            total_files += 1
    
    logger.info(f"📊 Found {total_files} total files")
    logger.info(f"📊 Found {len(samples)} unique sample IDs")
    
    # Check which samples are complete
    complete_samples = []
    incomplete_samples = []
    required_suffixes = ['-dvec.txt', '-mixed.wav', '-target.wav', '-mixed-features.pt', '-target-features.pt']
    
    for sample_id, files in samples.items():
        filenames = [os.path.basename(f) for f in files]
        
        # Check if all required files exist
        has_all_required = all(
            any(f.startswith(f"{sample_id}{suffix}") for f in filenames) 
            for suffix in required_suffixes
        )
        
        if has_all_required:
            complete_samples.append(sample_id)
        else:
            incomplete_samples.append(sample_id)
    
    logger.info(f"✅ Complete samples: {len(complete_samples)}")
    logger.info(f"❌ Incomplete samples: {len(incomplete_samples)}")
    
    if len(incomplete_samples) > 0:
        logger.warning(f"⚠️ First 10 incomplete samples: {incomplete_samples[:10]}")
    
    return samples, complete_samples

def estimate_subset_sizes(client, bucket_name, samples, complete_samples, target_size_gb=15):
    """Estimate how many samples fit in target size."""
    logger.info(f"📏 Estimating subset sizes for {target_size_gb}GB target...")
    
    # Sample a few complete samples to estimate average size
    sample_ids_to_check = complete_samples[:min(10, len(complete_samples))]
    total_size_bytes = 0
    files_checked = 0
    
    bucket = client.bucket(bucket_name)
    
    for sample_id in sample_ids_to_check:
        for file_path in samples[sample_id]:
            try:
                blob = bucket.blob(file_path)
                blob.reload()  # Get size info
                total_size_bytes += blob.size
                files_checked += 1
            except Exception as e:
                logger.warning(f"⚠️ Could not get size for {file_path}: {e}")
    
    if files_checked > 0:
        avg_size_per_file = total_size_bytes / files_checked
        avg_files_per_sample = files_checked / len(sample_ids_to_check)
        avg_size_per_sample = avg_size_per_file * avg_files_per_sample
        
        target_size_bytes = target_size_gb * 1024 * 1024 * 1024
        samples_per_subset = int(target_size_bytes / avg_size_per_sample)
        
        logger.info(f"📊 Average size per sample: {avg_size_per_sample/1024/1024:.1f} MB")
        logger.info(f"📊 Estimated samples per {target_size_gb}GB subset: {samples_per_subset}")
        
        return samples_per_subset
    else:
        logger.warning("⚠️ Could not estimate sizes, using default: 1000 samples per subset")
        return 1000

def create_subsets(client, bucket_name, samples, complete_samples, samples_per_subset, target_bucket, base_prefix, start_subset=1):
    """Create subsets by copying files to new GCS locations."""
    logger.info(f"🔄 Creating subsets with {samples_per_subset} samples each...")
    
    source_bucket = client.bucket(bucket_name)
    target_bucket_obj = client.bucket(target_bucket)
    
    num_subsets = (len(complete_samples) + samples_per_subset - 1) // samples_per_subset
    logger.info(f"📦 Will create {num_subsets} subsets (starting from subset {start_subset})")
    
    for subset_idx in range(start_subset - 1, num_subsets):
        start_idx = subset_idx * samples_per_subset
        end_idx = min(start_idx + samples_per_subset, len(complete_samples))
        subset_samples = complete_samples[start_idx:end_idx]
        
        logger.info(f"📦 Creating subset {subset_idx + 1}/{num_subsets} with {len(subset_samples)} samples...")
        
        subset_prefix = f"{base_prefix}/subset_{subset_idx + 1:02d}"
        
        files_copied = 0
        for sample_id in subset_samples:
            for source_file_path in samples[sample_id]:
                # Create target path
                filename = os.path.basename(source_file_path)
                target_file_path = f"{subset_prefix}/train/{filename}"
                
                try:
                    # Copy file
                    source_blob = source_bucket.blob(source_file_path)
                    target_blob = target_bucket_obj.blob(target_file_path)
                    
                    # Copy the blob
                    target_blob.rewrite(source_blob)
                    files_copied += 1
                    
                    if files_copied % 100 == 0:
                        logger.info(f"  📄 Copied {files_copied} files...")
                        
                except Exception as e:
                    logger.error(f"❌ Error copying {source_file_path}: {e}")
        
        logger.info(f"✅ Subset {subset_idx + 1} completed: {files_copied} files copied")
        
        # Create a metadata file for this subset
        metadata_content = f"""VoiceFilter Dataset Subset {subset_idx + 1}
Samples: {len(subset_samples)}
Files: {files_copied}
Sample IDs: {subset_samples[0]} to {subset_samples[-1]}
Created for Kaggle/limited storage training
"""
        metadata_blob = target_bucket_obj.blob(f"{subset_prefix}/README.txt")
        metadata_blob.upload_from_string(metadata_content)
    
    logger.info(f"🎉 All {num_subsets} subsets created successfully!")

def copy_test_data(client, source_bucket_name, source_test_prefix, target_bucket, target_prefix):
    """Copy test data (no subsetting needed)."""
    logger.info("📋 Copying test data...")
    
    source_bucket = client.bucket(source_bucket_name)
    target_bucket_obj = client.bucket(target_bucket)
    
    blobs = source_bucket.list_blobs(prefix=source_test_prefix)
    files_copied = 0
    
    for blob in blobs:
        if blob.name.endswith('/'):  # Skip directories
            continue
            
        filename = os.path.basename(blob.name)
        target_path = f"{target_prefix}/test/{filename}"
        
        try:
            target_blob = target_bucket_obj.blob(target_path)
            target_blob.rewrite(blob)
            files_copied += 1
            
            if files_copied % 50 == 0:
                logger.info(f"  📄 Copied {files_copied} test files...")
                
        except Exception as e:
            logger.error(f"❌ Error copying test file {blob.name}: {e}")
    
    logger.info(f"✅ Test data copied: {files_copied} files")

def main():
    parser = argparse.ArgumentParser(description='Create VoiceFilter dataset subsets in GCS')
    parser.add_argument('--source-bucket', required=True, help='Source GCS bucket name')
    parser.add_argument('--source-train-prefix', default='vf_data_production/train', 
                       help='Source training data prefix')
    parser.add_argument('--source-test-prefix', default='vf_data_production/test',
                       help='Source test data prefix') 
    parser.add_argument('--target-bucket', required=True, help='Target GCS bucket name')
    parser.add_argument('--target-prefix', default='vf_data_subsets', help='Target prefix for subsets')
    parser.add_argument('--subset-size-gb', type=int, default=15, help='Target size per subset in GB')
    parser.add_argument('--start-subset', type=int, default=1, help='Subset number to start from (for resuming)')
    parser.add_argument('--dry-run', action='store_true', help='Analyze only, do not copy files')
    
    args = parser.parse_args()
    
    logger.info("🚀 VoiceFilter Dataset Subset Creator")
    logger.info(f"📂 Source: gs://{args.source_bucket}/{args.source_train_prefix}")
    logger.info(f"📂 Target: gs://{args.target_bucket}/{args.target_prefix}")
    
    # Initialize GCS client
    client = storage.Client()
    
    # Step 1: Analyze dataset structure
    samples, complete_samples = analyze_dataset_structure(
        client, args.source_bucket, args.source_train_prefix
    )
    
    if len(complete_samples) == 0:
        logger.error("❌ No complete samples found!")
        return
    
    # Step 2: Estimate subset sizes
    samples_per_subset = estimate_subset_sizes(
        client, args.source_bucket, samples, complete_samples, args.subset_size_gb
    )
    
    if args.dry_run:
        logger.info("🏃 Dry run mode - analysis complete, no files copied")
        num_subsets = (len(complete_samples) + samples_per_subset - 1) // samples_per_subset
        logger.info(f"📊 Would create {num_subsets} subsets with ~{samples_per_subset} samples each")
        return
    
    # Step 3: Create subsets
    create_subsets(
        client, args.source_bucket, samples, complete_samples, 
        samples_per_subset, args.target_bucket, args.target_prefix, args.start_subset
    )
    
    # Step 4: Copy test data
    copy_test_data(
        client, args.source_bucket, args.source_test_prefix,
        args.target_bucket, args.target_prefix
    )
    
    logger.info("🎉 Dataset subset creation completed!")

if __name__ == '__main__':
    main() 