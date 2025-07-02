#!/usr/bin/env python3
"""
VoiceFilter-Lite 2020 Training Script for Google Colab
Optimized for mobile deployment and RTranslator integration
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("🚀 Running in Google Colab!")
except ImportError:
    IN_COLAB = False
    print("💻 Running locally")

def setup_colab_environment():
    """Setup Google Colab environment"""
    if not IN_COLAB:
        return
    
    print("📦 Setting up Colab anvironment...")
    
    # Install dependencies, including speechbrain for the new embedder
    os.system("pip install -q torch torchvision torchaudio")
    os.system("pip install -q librosa mir_eval pyyaml tensorboardX speechbrain")
    
    # Download LibriSpeech sample data (smaller subset for Colab)
    print("📥 Downloading LibriSpeech sample data...")
    os.system("wget -q http://www.openslr.org/resources/12/dev-clean.tar.gz")
    os.system("tar -xzf dev-clean.tar.gz")
    
    print("✅ Colab environment ready!")

def load_config():
    """Load VoiceFilter-Lite 2020 configuration"""
    config = {
        # Model architecture (VoiceFilter-Lite 2020)
        'model': {
            'input_dim': 384,  # 128 filterbanks × 3 stacked frames
            'mask_dim': 128,  # Mask output dimension
            'lstm_layers': 3,
            'lstm_dim': 512,
            'speaker_embed_dim': 256,
            'noise_classes': 2,  # clean/non-speech vs overlapped speech
        },
        
        # Training parameters
        'training': {
            'batch_size': 8 if IN_COLAB else 16,
            'learning_rate': 1e-4,
            'epochs': 50,
            'warmup_steps': 4000,
            'grad_clip': 5.0,
        },
        
        # Audio processing (VoiceFilter-Lite 2020)
        'audio': {
            'sample_rate': 16000,
            'num_filterbanks': 128,
            'frame_stack_size': 3,
            'frame_shift': 10,  # ms
            'frame_length': 25,  # ms
        },
        
        # Adaptive suppression
        'adaptive': {
            'alpha': 0.5,      # Mixing coefficient base
            'beta': 0.9,       # Exponential moving average
            'adaptation_rate': 0.1,
        },
        
        # Paths
        'paths': {
            'data_dir': './LibriSpeech/dev-clean' if IN_COLAB else './data',
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs',
        }
    }
    return config

class VoiceFilterLite2020Trainer:
    """Complete training pipeline for VoiceFilter-Lite 2020"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")
        
        # Create directories
        os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['paths']['log_dir'], exist_ok=True)
        
        # Initialize model, optimizer, etc.
        self.setup_model()
        self.setup_training()
        
    def setup_model(self):
        """Initialize VoiceFilter-Lite 2020 model"""
        from model.model import VoiceFilterLite2020
        from speechbrain.inference.speaker import EncoderClassifier

        self.model = VoiceFilterLite2020(
            input_dim=self.config['model']['input_dim'],
            lstm_layers=self.config['model']['lstm_layers'],
            lstm_dim=self.config['model']['lstm_dim'],
            speaker_embed_dim=self.config['model']['speaker_embed_dim'],
            num_noise_classes=self.config['model']['noise_classes']
        ).to(self.device)
        
        # Load multilingual pretrained speaker embedder from TalTechNLP/HuggingFace
        print("📎 Loading multilingual speaker embedder from TalTechNLP...")
        self.speaker_embedder = EncoderClassifier.from_hparams(
            source="TalTechNLP/voxlingua107-epaca-tdnn", 
            savedir="pretrained_models/lang-id-voxlingua107-ecapa",
            run_opts={"device": self.device}
        )
        # Freeze the speaker embedder's weights
        for param in self.speaker_embedder.parameters():
            param.requires_grad = False
        self.speaker_embedder.eval()
        
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_training(self):
        """Setup optimizer, scheduler, loss functions"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=1e-5
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['training']['learning_rate'],
            epochs=self.config['training']['epochs'],
            steps_per_epoch=100,  # Estimate
            pct_start=0.1
        )
        
        # Loss functions (VoiceFilter-Lite 2020)
        self.asymmetric_l2_loss = self.build_asymmetric_l2_loss()
        self.noise_classification_loss = nn.CrossEntropyLoss()
        
    def build_asymmetric_l2_loss(self):
        """Asymmetric L2 loss from VoiceFilter-Lite 2020 paper"""
        def asymmetric_l2_loss(enhanced, target, alpha=2.0):
            """
            Asymmetric L2 loss: heavier penalty for under-suppression
            Args:
                enhanced: Enhanced filterbank features
                target: Clean target features  
                alpha: Asymmetry factor (>1 penalizes under-suppression more)
            """
            diff = enhanced - target
            
            # Different weights for positive vs negative errors
            weights = torch.where(
                diff > 0,  # Over-suppression (enhanced < target)
                torch.ones_like(diff),  # Normal weight
                alpha * torch.ones_like(diff)  # Higher weight for under-suppression
            )
            
            return torch.mean(weights * diff ** 2)
        
        return asymmetric_l2_loss
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_mask_loss = 0
        total_noise_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Batch format: (mixed_features, clean_features, reference_audio, noise_type)
            mixed_features = batch['mixed_features'].to(self.device)
            clean_features = batch['clean_features'].to(self.device)
            reference_audio = batch['reference_audio'].to(self.device)
            noise_type = batch['noise_type'].to(self.device)
            
            # Get speaker embeddings
            with torch.no_grad():
                # The SpeechBrain model expects a tensor and its length.
                # We need to unsqueeze to add a batch dimension.
                speaker_embed = self.speaker_embedder.encode_batch(reference_audio)
                # The output has an extra dimension, so we squeeze it.
                speaker_embed = speaker_embed.squeeze(1)

            # Forward pass
            enhanced_features, noise_pred = self.model(mixed_features, speaker_embed)
            
            # Compute losses
            mask_loss = self.asymmetric_l2_loss(enhanced_features, clean_features)
            noise_loss = self.noise_classification_loss(noise_pred, noise_type)
            
            # Total loss (weighted combination)
            loss = mask_loss + 0.1 * noise_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['grad_clip']
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_mask_loss += mask_loss.item()
            total_noise_loss += noise_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Mask': f"{mask_loss.item():.4f}",
                'Noise': f"{noise_loss.item():.4f}",
                'LR': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
        
        return {
            'total_loss': total_loss / len(dataloader),
            'mask_loss': total_mask_loss / len(dataloader),
            'noise_loss': total_noise_loss / len(dataloader)
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['paths']['checkpoint_dir'], 
            f'voicefilter_lite_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config['paths']['checkpoint_dir'], 
                'voicefilter_lite_best.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model at epoch {epoch}")
    
    def train(self):
        """Complete training loop"""
        print("🚀 Starting VoiceFilter-Lite 2020 training...")
        
        # Create dummy dataloader for demonstration
        # In practice, use your actual DataLoader
        dummy_dataloader = self.create_dummy_dataloader()
        
        best_loss = float('inf')
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            print(f"\n📊 Epoch {epoch}/{self.config['training']['epochs']}")
            
            # Train epoch
            metrics = self.train_epoch(dummy_dataloader, epoch)
            
            # Check if best model
            is_best = metrics['total_loss'] < best_loss
            if is_best:
                best_loss = metrics['total_loss']
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, metrics, is_best)
            
            # Print epoch summary
            print(f"📈 Epoch {epoch} - Loss: {metrics['total_loss']:.4f}, "
                  f"Mask: {metrics['mask_loss']:.4f}, "
                  f"Noise: {metrics['noise_loss']:.4f}")
        
        print("✅ Training completed!")
        
        # Quantize model for mobile deployment
        self.quantize_for_mobile()
    
    def create_dummy_dataloader(self):
        """Create dummy dataloader for testing (replace with real data)"""
        class DummyDataset:
            def __len__(self):
                return 100  # Small dataset for Colab
            
            def __getitem__(self, idx):
                return {
                    'mixed_features': torch.randn(384, 100),  # 384D features, 100 frames
                    'clean_features': torch.randn(128, 100),  # 128D filterbank target
                    'reference_audio': torch.randn(16000),    # 1 second reference
                    'noise_type': torch.randint(0, 2, (1,)).item()  # Binary noise classification
                }
        
        dataset = DummyDataset()
        return DataLoader(dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
    
    def quantize_for_mobile(self):
        """Quantize model to 2.2MB for mobile deployment"""
        print("📱 Quantizing model for mobile deployment...")
        
        # Load best model
        best_path = os.path.join(self.config['paths']['checkpoint_dir'], 'voicefilter_lite_best.pt')
        checkpoint = torch.load(best_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, 
            {nn.Linear, nn.LSTM}, 
            dtype=torch.qint8
        )
        
        # Save quantized model
        quantized_path = os.path.join(
            self.config['paths']['checkpoint_dir'], 
            'voicefilter_lite_quantized.pt'
        )
        torch.save(quantized_model.state_dict(), quantized_path)
        
        # Check model size
        model_size = os.path.getsize(quantized_path) / (1024 * 1024)  # MB
        print(f"📦 Quantized model size: {model_size:.2f} MB (target: 2.2 MB)")
        
        if model_size <= 2.5:  # Some tolerance
            print("✅ Model size target achieved!")
        else:
            print("⚠️ Model size larger than target, consider further pruning")
        
        return quantized_path

def main():
    """Main training function"""
    print("🎯 VoiceFilter-Lite 2020 Training for RTranslator")
    print("=" * 60)
    
    # Setup environment
    if IN_COLAB:
        setup_colab_environment()
    
    # Load configuration
    config = load_config()
    
    # Initialize trainer
    trainer = VoiceFilterLite2020Trainer(config)
    
    # Start training
    trainer.train()
    
    print("\n🎉 VoiceFilter-Lite 2020 training completed!")
    print("Ready for RTranslator integration! 🚀")

if __name__ == "__main__":
    main() 