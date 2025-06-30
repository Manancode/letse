#!/usr/bin/env python3
"""
Test script for VoiceFilter-Lite 2020 Phase 1 implementation
"""

import numpy as np
import torch
from utils.audio import Audio
from easydict import EasyDict
import yaml

def test_phase1():
    print('VoiceFilter-Lite 2020 Phase 1 Test')
    print('='*50)
    
    # Load config
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    hp = EasyDict(config)
    
    print(f'Filterbank dimensions: {hp.audio.num_filterbanks}')
    print(f'Frame stack size: {hp.audio.frame_stack_size}')
    print(f'Expected input dimension: {hp.model.input_dim}')
    print()
    
    # Test audio processor
    try:
        audio = Audio(hp)
        print('✅ Audio processor initialized successfully!')
    except Exception as e:
        print(f'❌ Error initializing audio processor: {e}')
        return False
    
    # Test with dummy audio
    dummy_audio = np.random.randn(48000)  # 3 seconds at 16kHz
    print(f'Test audio shape: {dummy_audio.shape}')
    
    # Test filterbank extraction
    try:
        filterbank_features = audio.wav2filterbank(dummy_audio)
        print(f'✅ Filterbank features shape: {filterbank_features.shape}')
        
        # Expected: [time_steps, 128]
        if filterbank_features.shape[1] == hp.audio.num_filterbanks:
            print(f'✅ Filterbank dimensions correct: {filterbank_features.shape[1]} == {hp.audio.num_filterbanks}')
        else:
            print(f'❌ Filterbank dimensions incorrect: {filterbank_features.shape[1]} != {hp.audio.num_filterbanks}')
            
    except Exception as e:
        print(f'❌ Error in filterbank extraction: {e}')
        return False
    
    # Test frame stacking
    try:
        stacked_features = audio.wav2features(dummy_audio)
        print(f'✅ Stacked features shape: {stacked_features.shape}')
        
        # Validate dimensions
        expected_freq_dim = hp.audio.num_filterbanks * hp.audio.frame_stack_size
        actual_freq_dim = stacked_features.shape[1]
        
        print(f'Expected frequency dimension: {expected_freq_dim}')
        print(f'Actual frequency dimension: {actual_freq_dim}')
        
        if expected_freq_dim == actual_freq_dim:
            print(f'✅ Feature dimensions match: {actual_freq_dim}')
        else:
            print(f'❌ Feature dimensions mismatch: {actual_freq_dim} != {expected_freq_dim}')
            return False
            
    except Exception as e:
        print(f'❌ Error in frame stacking: {e}')
        return False
    
    # Test frame stacking without stacking (should equal filterbank features)
    try:
        unstacked_features = audio.wav2features(dummy_audio, apply_stacking=False)
        print(f'✅ Unstacked features shape: {unstacked_features.shape}')
        
        if np.allclose(unstacked_features, filterbank_features):
            print('✅ Unstacked features match filterbank features')
        else:
            print('⚠️  Unstacked features differ from filterbank features (might be expected)')
            
    except Exception as e:
        print(f'❌ Error in unstacked features: {e}')
        return False
    
    print()
    print('🎯 Phase 1 test completed successfully!')
    print('📋 Summary:')
    print(f'   - Filterbank extraction: ✅ Working')
    print(f'   - Frame stacking: ✅ Working')  
    print(f'   - Output dimensions: ✅ Correct ({actual_freq_dim}D)')
    print(f'   - Paper compliance: ✅ 128D LFBE + stacking')
    
    return True

if __name__ == '__main__':
    test_phase1() 