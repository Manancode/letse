#!/usr/bin/env python3
"""
VoiceFilter-Lite 2020 Model Quantization
Target: 2.2MB model size (Google Slides conclusion)

Implements quantization techniques to reduce model size while maintaining performance:
- Post-training quantization (PTQ)
- Dynamic quantization  
- Model pruning
- Mobile deployment optimization
"""

import torch
import torch.quantization as quantization
import torch.nn.utils.prune as prune
import os
import argparse
from pathlib import Path
import numpy as np

from model.model import VoiceFilterLite
from utils.hparams import HParam


class VoiceFilterLiteQuantizer:
    """
    Google Slides Target: 2.2MB quantized model
    
    Implements various quantization and compression techniques:
    1. Dynamic quantization (int8)
    2. Post-training quantization  
    3. Model pruning
    4. Mobile-optimized deployment
    """
    
    def __init__(self, hp):
        self.hp = hp
        
    def quantize_dynamic(self, model: VoiceFilterLite, output_path: str) -> torch.jit.ScriptModule:
        """
        Dynamic quantization for inference speedup and size reduction
        Google slides target: 2.2MB
        """
        print("Applying dynamic quantization...")
        
        # Set model to evaluation mode
        model.eval()
        
        # Apply dynamic quantization to linear layers
        quantized_model = quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM, torch.nn.Conv1d},  # Quantize these layer types
            dtype=torch.qint8  # Use int8 quantization
        )
        
        # Calculate size reduction
        original_size = self._get_model_size(model)
        quantized_size = self._get_model_size(quantized_model)
        compression_ratio = original_size / quantized_size
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), output_path)
        
        return quantized_model
    
    def apply_pruning(self, model: VoiceFilterLite, pruning_ratio: float = 0.3) -> VoiceFilterLite:
        """
        Structured pruning to reduce model parameters
        Target: Additional size reduction while maintaining performance
        """
        print(f"Applying {pruning_ratio*100:.1f}% structured pruning...")
        
        # Prune LSTM layers (most parameters)
        for lstm_layer in model.lstm_stack:
            # Prune input-hidden weights
            prune.ln_structured(
                lstm_layer, 
                name='weight_ih_l0',
                amount=pruning_ratio,
                n=2,  # L2 norm
                dim=0  # Prune output channels
            )
            
            # Prune hidden-hidden weights  
            prune.ln_structured(
                lstm_layer,
                name='weight_hh_l0', 
                amount=pruning_ratio,
                n=2,
                dim=0
            )
        
        # Prune linear layers in heads
        prune.ln_structured(
            model.mask_head[0],
            name='weight',
            amount=pruning_ratio,
            n=2,
            dim=0
        )
        
        # Prune noise predictor
        for layer in model.noise_predictor:
            if isinstance(layer, torch.nn.Linear):
                prune.ln_structured(
                    layer,
                    name='weight', 
                    amount=pruning_ratio,
                    n=2,
                    dim=0
                )
        
        return model
    
    def post_training_quantization(self, model: VoiceFilterLite, calibration_data: torch.Tensor,
                                 output_path: str) -> torch.jit.ScriptModule:
        """
        Post-training quantization with calibration data
        More accurate than dynamic quantization
        """
        print("Applying post-training quantization...")
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        
        # Fuse operations for better quantization
        model_fused = self._fuse_model(model)
        
        # Prepare for quantization
        model_prepared = quantization.prepare(model_fused)
        
        # Calibrate with sample data
        print("Calibrating with sample data...")
        with torch.no_grad():
            for i in range(min(100, calibration_data.shape[0])):
                batch_features = calibration_data[i:i+1]
                dummy_dvec = torch.randn(1, self.hp.embedder.emb_dim)
                model_prepared(batch_features, dummy_dvec)
        
        # Convert to quantized model
        model_quantized = quantization.convert(model_prepared)
        
        # Calculate size
        quantized_size = self._get_model_size(model_quantized)
        print(f"Post-training quantized size: {quantized_size:.2f} MB")
        
        # Save
        torch.save(model_quantized.state_dict(), output_path)
        
        return model_quantized
    
    def create_mobile_optimized_model(self, model: VoiceFilterLite, 
                                    sample_input: tuple, output_path: str) -> torch.jit.ScriptModule:
        """
        Create mobile-optimized TorchScript model
        Google slides: Streaming, on-device deployment
        """
        print("Creating mobile-optimized model...")
        
        model.eval()
        
        # Trace model for mobile deployment
        features, dvec = sample_input
        traced_model = torch.jit.trace(model, (features, dvec))
        
        # Optimize for mobile
        traced_model_optimized = torch.jit.optimize_for_inference(traced_model)
        
        # Save mobile model
        traced_model_optimized.save(output_path)
        
        # Calculate final size
        model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Mobile-optimized model size: {model_size_mb:.2f} MB")
        
        return traced_model_optimized
    
    def achieve_google_slides_target(self, model: VoiceFilterLite, 
                                   calibration_data: torch.Tensor,
                                   output_dir: str) -> dict:
        """
        Complete pipeline to achieve Google slides 2.2MB target
        """
        print("VoiceFilter-Lite 2020 Quantization Pipeline")
        print("Google Slides Target: 2.2MB model size")
        print("=" * 50)
        
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # Original model size
        original_size = self._get_model_size(model)
        results['original_size_mb'] = original_size
        print(f"Original model size: {original_size:.2f} MB")
        
        # Step 1: Dynamic quantization
        dynamic_path = os.path.join(output_dir, 'voicefilter_lite_dynamic_quantized.pt')
        dynamic_model = self.quantize_dynamic(model, dynamic_path)
        dynamic_size = self._get_model_size(dynamic_model)
        results['dynamic_quantized_size_mb'] = dynamic_size
        
        # Step 2: Apply pruning
        model_pruned = self.apply_pruning(model.clone() if hasattr(model, 'clone') else model, 0.2)
        pruned_size = self._get_model_size(model_pruned)
        results['pruned_size_mb'] = pruned_size
        print(f"Pruned model size: {pruned_size:.2f} MB")
        
        # Step 3: Combined pruning + quantization
        combined_path = os.path.join(output_dir, 'voicefilter_lite_pruned_quantized.pt')
        combined_model = self.quantize_dynamic(model_pruned, combined_path)
        combined_size = self._get_model_size(combined_model)
        results['combined_size_mb'] = combined_size
        
        # Step 4: Mobile optimization
        sample_features = torch.randn(1, 100, self.hp.model.input_dim)
        sample_dvec = torch.randn(1, self.hp.embedder.emb_dim)
        mobile_path = os.path.join(output_dir, 'voicefilter_lite_mobile.pt')
        
        try:
            mobile_model = self.create_mobile_optimized_model(
                combined_model, (sample_features, sample_dvec), mobile_path)
            mobile_size = os.path.getsize(mobile_path) / (1024 * 1024)
            results['mobile_optimized_size_mb'] = mobile_size
        except Exception as e:
            print(f"Mobile optimization failed: {e}")
            results['mobile_optimized_size_mb'] = combined_size
        
        # Summary
        print("\n" + "=" * 50)
        print("QUANTIZATION RESULTS SUMMARY")
        print("=" * 50)
        print(f"Original size:           {original_size:.2f} MB")
        print(f"Dynamic quantized:       {dynamic_size:.2f} MB ({original_size/dynamic_size:.1f}x smaller)")
        print(f"Pruned:                  {pruned_size:.2f} MB ({original_size/pruned_size:.1f}x smaller)")
        print(f"Pruned + Quantized:      {combined_size:.2f} MB ({original_size/combined_size:.1f}x smaller)")
        print(f"Mobile optimized:        {results.get('mobile_optimized_size_mb', combined_size):.2f} MB")
        
        # Check if we achieved Google slides target
        final_size = results.get('mobile_optimized_size_mb', combined_size)
        target_size = 2.2
        
        if final_size <= target_size:
            print(f"\n✅ SUCCESS: Achieved Google slides target of {target_size} MB!")
            print(f"   Final size: {final_size:.2f} MB")
        else:
            print(f"\n⚠️  Close to target: {final_size:.2f} MB vs {target_size} MB target")
            print(f"   Additional optimization may be needed")
            
        results['achieved_target'] = final_size <= target_size
        results['target_size_mb'] = target_size
        
        return results
    
    def _get_model_size(self, model) -> float:
        """Calculate model size in MB"""
        if hasattr(model, 'state_dict'):
            # Standard PyTorch model
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            size_mb = (param_size + buffer_size) / (1024 * 1024)
        else:
            # Quantized or other model type
            # Rough estimate - in practice you'd save and check file size
            try:
                temp_path = '/tmp/temp_model.pt'
                torch.save(model, temp_path)
                size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                os.remove(temp_path)
            except:
                size_mb = 0.0
        
        return size_mb
    
    def _fuse_model(self, model: VoiceFilterLite) -> VoiceFilterLite:
        """Fuse operations for better quantization"""
        # In practice, you'd fuse Conv+ReLU, Linear+ReLU etc.
        # For demo, return the original model
        return model


def main():
    parser = argparse.ArgumentParser(description='VoiceFilter-Lite 2020 Model Quantization')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained VoiceFilter-Lite model')
    parser.add_argument('--output_dir', type=str, default='models/quantized',
                       help='Output directory for quantized models')
    parser.add_argument('--calibration_data', type=str,
                       help='Path to calibration data for post-training quantization')
    
    args = parser.parse_args()
    
    # Load configuration
    hp = HParam(args.config)
    
    print("VoiceFilter-Lite 2020 Model Quantization")
    print("Google Slides Target: 2.2MB")
    print("=" * 40)
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = VoiceFilterLite(hp)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully")
    else:
        print("⚠️  Model file not found, using randomly initialized model for demo")
    
    # Initialize quantizer
    quantizer = VoiceFilterLiteQuantizer(hp)
    
    # Load calibration data if provided
    calibration_data = None
    if args.calibration_data and os.path.exists(args.calibration_data):
        calibration_data = torch.load(args.calibration_data)
        print(f"Loaded calibration data: {calibration_data.shape}")
    else:
        # Generate dummy calibration data
        calibration_data = torch.randn(100, 100, hp.model.input_dim)
        print("Using dummy calibration data")
    
    # Run complete quantization pipeline
    results = quantizer.achieve_google_slides_target(
        model, calibration_data, args.output_dir)
    
    print(f"\nQuantized models saved to: {args.output_dir}")
    
    # Display Google slides context
    print("\n" + "=" * 60)
    print("GOOGLE SLIDES CONTEXT")
    print("=" * 60)
    print("Target specifications:")
    print("- Model size: 2.2 MB (quantized)")
    print("- Deployment: On-device, streaming")
    print("- Performance: No WER degradation on clean/non-speech")
    print("- Improvement: 25.1% WER improvement on overlapped speech")
    print("- Architecture: Tiny, fast, streaming, part of on-device ASR")


if __name__ == '__main__':
    main() 