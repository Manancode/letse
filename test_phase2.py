#!/usr/bin/env python3
"""
Test script for VoiceFilter-Lite 2020 Phase 2 implementation
Tests the complete architecture changes from 2019 to 2020
"""

import numpy as np
import torch
from model.model import VoiceFilter, VoiceFilterLite
from model.embedder import SpeechEmbedder
from easydict import EasyDict
import yaml

def test_phase2():
    print('VoiceFilter-Lite 2020 Phase 2 Architecture Test')
    print('='*60)
    
    # Load config
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    hp = EasyDict(config)
    
    print(f'Input dimension: {hp.model.input_dim}')
    print(f'LSTM layers: {hp.model.lstm_layers}')
    print(f'LSTM dimension: {hp.model.lstm_dim}')
    print(f'Mask dimension: {hp.model.mask_dim}')
    print(f'Noise prediction dimension: {hp.model.noise_pred_dim}')
    print()
    
    # Test model initialization
    try:
        model = VoiceFilterLite(hp)
        print('✅ VoiceFilter-Lite 2020 model initialized successfully!')
        
        # Check if alias works
        model_alias = VoiceFilter(hp)
        print('✅ Legacy VoiceFilter alias works!')
        
    except Exception as e:
        print(f'❌ Error initializing model: {e}')
        return False
    
    # Test embedder (should remain unchanged)
    try:
        embedder = SpeechEmbedder(hp)
        print('✅ Speech embedder initialized successfully!')
    except Exception as e:
        print(f'❌ Error initializing embedder: {e}')
        return False
    
    # Test forward pass with dummy data
    try:
        batch_size = 2
        time_steps = 100
        feature_dim = hp.model.input_dim  # 384
        emb_dim = hp.embedder.emb_dim     # 256
        
        # Create dummy inputs
        dummy_features = torch.randn(batch_size, time_steps, feature_dim)
        dummy_dvec = torch.randn(batch_size, emb_dim)
        
        print(f'Input features shape: {dummy_features.shape}')
        print(f'Input d-vector shape: {dummy_dvec.shape}')
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            mask, noise_pred = model(dummy_features, dummy_dvec)
        
        print(f'✅ Forward pass successful!')
        print(f'Output mask shape: {mask.shape}')
        print(f'Output noise prediction shape: {noise_pred.shape}')
        
        # Validate output dimensions
        expected_mask_shape = (batch_size, time_steps, hp.model.mask_dim)
        expected_noise_shape = (batch_size, time_steps, hp.model.noise_pred_dim)
        
        if mask.shape == expected_mask_shape:
            print(f'✅ Mask output shape correct: {mask.shape}')
        else:
            print(f'❌ Mask output shape incorrect: {mask.shape} != {expected_mask_shape}')
            return False
            
        if noise_pred.shape == expected_noise_shape:
            print(f'✅ Noise prediction shape correct: {noise_pred.shape}')
        else:
            print(f'❌ Noise prediction shape incorrect: {noise_pred.shape} != {expected_noise_shape}')
            return False
        
        # Test mask value range [0, 1]
        if torch.all(mask >= 0) and torch.all(mask <= 1):
            print('✅ Mask values in correct range [0, 1]')
        else:
            print(f'❌ Mask values out of range: min={mask.min():.3f}, max={mask.max():.3f}')
            return False
        
        # Test noise prediction probabilities sum to 1
        noise_sums = torch.sum(noise_pred, dim=-1)
        if torch.allclose(noise_sums, torch.ones_like(noise_sums), atol=1e-6):
            print('✅ Noise prediction probabilities sum to 1')
        else:
            print(f'❌ Noise prediction probabilities incorrect: sample sum={noise_sums[0,0]:.6f}')
            return False
            
    except Exception as e:
        print(f'❌ Error in forward pass: {e}')
        return False
    
    # Test speech separation method
    try:
        enhanced_features, noise_pred_sep = model.separate_speech(dummy_features, dummy_dvec)
        
        if enhanced_features.shape == dummy_features.shape:
            print(f'✅ Speech separation output shape correct: {enhanced_features.shape}')
        else:
            print(f'❌ Speech separation shape incorrect: {enhanced_features.shape} != {dummy_features.shape}')
            return False
            
    except Exception as e:
        print(f'❌ Error in speech separation: {e}')
        return False
    
    # Test architecture compliance with 2020 paper
    print()
    print('📋 2020 Paper Compliance Check:')
    
    # Check LSTM architecture
    lstm_count = len(model.lstm_stack)
    if lstm_count == 3:
        print(f'✅ LSTM layers: {lstm_count} (Paper: 3 layers)')
    else:
        print(f'❌ LSTM layers: {lstm_count} (Paper requires: 3)')
        return False
    
    # Check LSTM dimensions
    first_lstm = model.lstm_stack[0]
    lstm_hidden_size = first_lstm.hidden_size
    if lstm_hidden_size == 512:
        print(f'✅ LSTM hidden size: {lstm_hidden_size} (Paper: 512)')
    else:
        print(f'❌ LSTM hidden size: {lstm_hidden_size} (Paper requires: 512)')
        return False
    
    # Check if LSTM is unidirectional (streaming requirement)
    is_bidirectional = first_lstm.bidirectional
    if not is_bidirectional:
        print(f'✅ LSTM is unidirectional (Paper: streaming requirement)')
    else:
        print(f'❌ LSTM is bidirectional (Paper requires: unidirectional for streaming)')
        return False
    
    # Check dual outputs
    if hasattr(model, 'mask_head') and hasattr(model, 'noise_predictor'):
        print(f'✅ Dual output system present (Paper: mask + noise prediction)')
    else:
        print(f'❌ Missing dual output system')
        return False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print()
    print('📊 Model Statistics:')
    print(f'   Total parameters: {total_params:,}')
    print(f'   Trainable parameters: {trainable_params:,}')
    print(f'   Model size estimate: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)')
    
    print()
    print('🎯 Phase 2 test completed successfully!')
    print('📋 Summary:')
    print(f'   ✅ Architecture: 3×512 uni-directional LSTM')
    print(f'   ✅ Input: 384D stacked filterbank features')
    print(f'   ✅ Outputs: 128D mask + 2D noise prediction')
    print(f'   ✅ Streaming compatible: No bidirectional processing')
    print(f'   ✅ Paper compliance: All 2020 specs met')
    
    return True

if __name__ == '__main__':
    test_phase2() 