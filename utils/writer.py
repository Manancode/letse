import numpy as np
from tensorboardX import SummaryWriter

from .plotting import plot_spectrogram_to_numpy


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_evaluation_2019(self, test_loss, sdr,
                       mixed_wav, target_wav, est_wav,
                       mixed_spec, target_spec, est_spec, est_mask,
                       step):
        """Legacy VoiceFilter 2019 evaluation logging"""
        
        self.add_scalar('test_loss', test_loss, step)
        self.add_scalar('SDR', sdr, step)

        self.add_audio('mixed_wav', mixed_wav, step, self.hp.audio.sample_rate)
        self.add_audio('target_wav', target_wav, step, self.hp.audio.sample_rate)
        self.add_audio('estimated_wav', est_wav, step, self.hp.audio.sample_rate)

        self.add_image('data/mixed_spectrogram',
            plot_spectrogram_to_numpy(mixed_spec), step, dataformats='HWC')
        self.add_image('data/target_spectrogram',
            plot_spectrogram_to_numpy(target_spec), step, dataformats='HWC')
        self.add_image('result/estimated_spectrogram',
            plot_spectrogram_to_numpy(est_spec), step, dataformats='HWC')
        self.add_image('result/estimated_mask',
            plot_spectrogram_to_numpy(est_mask), step, dataformats='HWC')
        self.add_image('result/estimation_error_sq',
            plot_spectrogram_to_numpy(np.square(est_spec - target_spec)), step, dataformats='HWC')

    def log_evaluation_2020(self, test_loss, mask_loss, noise_loss, wer,
                           mixed_features, target_features, enhanced_features, 
                           soft_mask, noise_predictions, adaptive_weights,
                           step):
        """VoiceFilter-Lite 2020 evaluation logging"""
        
        # Loss metrics
        self.add_scalar('test_loss', test_loss, step)
        self.add_scalar('mask_loss', mask_loss, step)
        self.add_scalar('noise_classification_loss', noise_loss, step)
        
        # Performance metrics (WER instead of SDR)
        self.add_scalar('WER', wer, step)
        
        # Noise type classification accuracy
        if noise_predictions is not None:
            noise_accuracy = self._compute_noise_accuracy(noise_predictions)
            self.add_scalar('noise_classification_accuracy', noise_accuracy, step)
        
        # Adaptive suppression metrics
        if adaptive_weights is not None:
            self.add_scalar('adaptive_weight_mean', np.mean(adaptive_weights), step)
            self.add_scalar('adaptive_weight_std', np.std(adaptive_weights), step)
            self.add_histogram('adaptive_weights_distribution', adaptive_weights, step)
        
        # Filterbank feature visualizations
        if mixed_features is not None:
            self.add_image('data/mixed_filterbank',
                plot_spectrogram_to_numpy(mixed_features), step, dataformats='HWC')
        
        if target_features is not None:
            self.add_image('data/target_filterbank',
                plot_spectrogram_to_numpy(target_features), step, dataformats='HWC')
        
        if enhanced_features is not None:
            self.add_image('result/enhanced_filterbank',
                plot_spectrogram_to_numpy(enhanced_features), step, dataformats='HWC')
        
        if soft_mask is not None:
            self.add_image('result/soft_mask',
                plot_spectrogram_to_numpy(soft_mask), step, dataformats='HWC')
        
        # Feature enhancement quality
        if target_features is not None and enhanced_features is not None:
            enhancement_error = np.square(enhanced_features - target_features)
            self.add_image('result/enhancement_error_sq',
                plot_spectrogram_to_numpy(enhancement_error), step, dataformats='HWC')
            
            # Mean squared error per frequency band
            mse_per_band = np.mean(enhancement_error, axis=1)  # Average over time
            self.add_histogram('enhancement_mse_per_frequency_band', mse_per_band, step)

    def log_evaluation(self, test_loss, *args, step=None, **kwargs):
        """
        Unified evaluation logging that detects VoiceFilter version
        """
        if step is None:
            raise ValueError("step parameter is required")
        
        # Detect if this is 2019 or 2020 based on arguments
        if len(args) >= 6 and 'sdr' not in kwargs:
            # VoiceFilter 2019: (test_loss, sdr, mixed_wav, target_wav, est_wav, mixed_spec, target_spec, est_spec, est_mask)
            sdr = args[0]
            mixed_wav = args[1] 
            target_wav = args[2]
            est_wav = args[3]
            mixed_spec = args[4]
            target_spec = args[5] 
            est_spec = args[6]
            est_mask = args[7]
            
            self.log_evaluation_2019(test_loss, sdr, mixed_wav, target_wav, est_wav,
                                    mixed_spec, target_spec, est_spec, est_mask, step)
        
        elif 'wer' in kwargs or 'mask_loss' in kwargs:
            # VoiceFilter-Lite 2020: keyword arguments
            mask_loss = kwargs.get('mask_loss', 0.0)
            noise_loss = kwargs.get('noise_loss', 0.0)
            wer = kwargs.get('wer', 0.0)
            mixed_features = kwargs.get('mixed_features', None)
            target_features = kwargs.get('target_features', None)
            enhanced_features = kwargs.get('enhanced_features', None)
            soft_mask = kwargs.get('soft_mask', None)
            noise_predictions = kwargs.get('noise_predictions', None)
            adaptive_weights = kwargs.get('adaptive_weights', None)
            
            self.log_evaluation_2020(test_loss, mask_loss, noise_loss, wer,
                                    mixed_features, target_features, enhanced_features,
                                    soft_mask, noise_predictions, adaptive_weights, step)
        
        else:
            # Fallback: basic logging
            self.add_scalar('test_loss', test_loss, step)
            print(f"⚠️ Unknown evaluation format, logged test_loss only")

    def _compute_noise_accuracy(self, noise_predictions):
        """Compute noise type classification accuracy"""
        if len(noise_predictions) == 0:
            return 0.0
        
        # Assuming noise_predictions is a list of (predicted, actual) tuples
        correct = sum(1 for pred, actual in noise_predictions if pred == actual)
        return correct / len(noise_predictions)

    def log_training_2020(self, train_loss, mask_loss, noise_loss, step):
        """Enhanced training logging for VoiceFilter-Lite 2020"""
        self.add_scalar('train_loss', train_loss, step)
        self.add_scalar('train_mask_loss', mask_loss, step)
        self.add_scalar('train_noise_loss', noise_loss, step)

    def log_model_info(self, model, input_shape, step=0):
        """Log model architecture information"""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.add_scalar('model/total_parameters', total_params, step)
            self.add_scalar('model/trainable_parameters', trainable_params, step)
            
            # Log parameter distribution
            all_params = []
            for param in model.parameters():
                all_params.extend(param.detach().cpu().numpy().flatten())
            
            self.add_histogram('model/parameter_distribution', np.array(all_params), step)
            
            print(f"📊 Model info logged: {total_params:,} total params, {trainable_params:,} trainable")
            
        except Exception as e:
            print(f"⚠️ Could not log model info: {e}")

    def log_learning_rate(self, lr, step):
        """Log current learning rate"""
        self.add_scalar('training/learning_rate', lr, step)

    def log_gradient_norm(self, model, step):
        """Log gradient norms for debugging"""
        try:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            self.add_scalar('training/gradient_norm', total_norm, step)
            
        except Exception as e:
            print(f"⚠️ Could not log gradient norm: {e}")

    def log_audio_samples_2020(self, mixed_audio, target_audio, enhanced_audio, step, sample_rate=16000):
        """Log audio samples for VoiceFilter-Lite 2020"""
        self.add_audio('audio/mixed', mixed_audio, step, sample_rate)
        self.add_audio('audio/target', target_audio, step, sample_rate)
        self.add_audio('audio/enhanced', enhanced_audio, step, sample_rate)
