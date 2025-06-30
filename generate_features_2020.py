#!/usr/bin/env python3
"""
VoiceFilter-Lite 2020 Feature Generation Script

This script converts existing audio files to 128D stacked filterbank features
as required by the VoiceFilter-Lite 2020 paper.

Usage:
    python generate_features_2020.py --data_dir /path/to/data --config config/default.yaml
"""

import os
import glob
import argparse
import torch
import librosa
import numpy as np
from tqdm import tqdm
import yaml
from easydict import EasyDict

from utils.audio import Audio


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


def generate_features_for_directory(data_dir, hp, audio_processor):
    """
    Generate VoiceFilter-Lite 2020 filterbank features for all audio files
    """
    print(f"Processing directory: {data_dir}")
    
    # Find all target and mixed audio files
    target_wav_files = sorted(glob.glob(os.path.join(data_dir, hp.form.target.wav)))
    mixed_wav_files = sorted(glob.glob(os.path.join(data_dir, hp.form.mixed.wav)))
    
    print(f"Found {len(target_wav_files)} target files")
    print(f"Found {len(mixed_wav_files)} mixed files")
    
    assert len(target_wav_files) == len(mixed_wav_files), \
        "Number of target and mixed files must match"
    
    # Process target files
    print("Generating target features...")
    for target_wav_path in tqdm(target_wav_files):
        try:
            # Load audio
            wav, _ = librosa.load(target_wav_path, sr=hp.audio.sample_rate)
            
            # Extract filterbank features
            features = audio_processor.wav2features(wav)
            
            # Convert to torch tensor
            features_tensor = torch.from_numpy(features).float()
            
            # Generate output path
            target_features_path = target_wav_path.replace('-target.wav', '-target-features.pt')
            
            # Save features
            torch.save(features_tensor, target_features_path)
            
        except Exception as e:
            print(f"Error processing {target_wav_path}: {e}")
            continue
    
    # Process mixed files  
    print("Generating mixed features...")
    for mixed_wav_path in tqdm(mixed_wav_files):
        try:
            # Load audio
            wav, _ = librosa.load(mixed_wav_path, sr=hp.audio.sample_rate)
            
            # Extract filterbank features
            features = audio_processor.wav2features(wav)
            
            # Convert to torch tensor
            features_tensor = torch.from_numpy(features).float()
            
            # Generate output path
            mixed_features_path = mixed_wav_path.replace('-mixed.wav', '-mixed-features.pt')
            
            # Save features
            torch.save(features_tensor, mixed_features_path)
            
        except Exception as e:
            print(f"Error processing {mixed_wav_path}: {e}")
            continue
    
    print(f"Feature generation completed for {data_dir}")


def validate_features(data_dir, hp):
    """
    Validate generated features against paper specifications
    """
    print("Validating generated features...")
    
    # Find a few feature files to validate
    target_features_files = glob.glob(os.path.join(data_dir, '*-target-features.pt'))[:5]
    mixed_features_files = glob.glob(os.path.join(data_dir, '*-mixed-features.pt'))[:5]
    
    for features_file in target_features_files + mixed_features_files:
        try:
            features = torch.load(features_file)
            time_steps, feature_dim = features.shape
            
            expected_dim = hp.model.input_dim  # Should be 384 (128 * 3)
            
            print(f"File: {os.path.basename(features_file)}")
            print(f"  Shape: {features.shape}")
            print(f"  Expected feature dim: {expected_dim}")
            print(f"  Actual feature dim: {feature_dim}")
            print(f"  Valid: {feature_dim == expected_dim}")
            print()
            
            if feature_dim != expected_dim:
                print(f"WARNING: Feature dimension mismatch in {features_file}")
                
        except Exception as e:
            print(f"Error validating {features_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate VoiceFilter-Lite 2020 features')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--validate', action='store_true',
                        help='Validate generated features')
    
    args = parser.parse_args()
    
    # Load configuration
    hp = load_config(args.config)
    
    # Initialize audio processor
    audio_processor = Audio(hp)
    
    print("VoiceFilter-Lite 2020 Feature Generation")
    print("=" * 50)
    print(f"Configuration: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Filterbank dimensions: {hp.audio.num_filterbanks}")
    print(f"Frame stack size: {hp.audio.frame_stack_size}")
    print(f"Expected input dimension: {hp.model.input_dim}")
    print("=" * 50)
    
    # Generate features
    if os.path.isdir(args.data_dir):
        generate_features_for_directory(args.data_dir, hp, audio_processor)
    else:
        print(f"Error: {args.data_dir} is not a valid directory")
        return
    
    # Validate features if requested
    if args.validate:
        validate_features(args.data_dir, hp)
    
    print("Done!")


if __name__ == '__main__':
    main() 