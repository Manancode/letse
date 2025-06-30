# adapted from Keith Ito's tacotron implementation
# https://github.com/keithito/tacotron/blob/master/util/audio.py
# Modified for VoiceFilter-Lite 2020: Filterbank features instead of spectrograms

import librosa
import numpy as np
from scipy.signal import get_window


class Audio():
    def __init__(self, hp):
        self.hp = hp
        # Mel basis for d-vector embedder (unchanged from 2019)
        self.mel_basis = librosa.filters.mel(sr=hp.audio.sample_rate,
                                             n_fft=hp.embedder.n_fft,
                                             n_mels=hp.embedder.num_mels)
        
        # Filterbank basis for VoiceFilter-Lite 2020
        # PAPER SPEC: 128-dimensional log filterbank energies (LFBE)
        self.filterbank_basis = librosa.filters.mel(sr=hp.audio.sample_rate,
                                                   n_fft=hp.audio.n_fft,
                                                   n_mels=hp.audio.num_filterbanks,
                                                   fmin=0.0,
                                                   fmax=hp.audio.sample_rate//2)

    def get_mel(self, y):
        """Keep original mel extraction for d-vector embedder (unchanged)"""
        y = librosa.core.stft(y=y, n_fft=self.hp.embedder.n_fft,
                              hop_length=self.hp.audio.hop_length,
                              win_length=self.hp.audio.win_length,
                              window='hann')
        magnitudes = np.abs(y) ** 2
        mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
        return mel

    def wav2filterbank(self, y):
        """
        VoiceFilter-Lite 2020: Convert audio to 128D log filterbank energies
        Paper: "stacked filterbank energies" as input features
        """
        # Compute STFT for filterbank extraction
        D = librosa.stft(y=y, n_fft=self.hp.audio.n_fft,
                        hop_length=self.hp.audio.hop_length,
                        win_length=self.hp.audio.win_length,
                        window='hann')
        
        # Magnitude spectrogram  
        magnitude = np.abs(D) ** 2
        
        # Apply filterbank
        filterbank = np.dot(self.filterbank_basis, magnitude)
        
        # Log filterbank energies (LFBE)
        log_filterbank = np.log(filterbank + 1e-8)
        
        # Transpose to [time, freq] format for consistency
        log_filterbank = log_filterbank.T
        
        return log_filterbank

    def stack_features(self, features, stack_size=None):
        """
        VoiceFilter-Lite 2020: Stack frames for temporal context
        Paper mentions "stacked filterbank energies"
        """
        if stack_size is None:
            stack_size = self.hp.audio.frame_stack_size
            
        if stack_size == 1:
            return features
            
        # Pad features for stacking
        pad_width = stack_size // 2
        padded_features = np.pad(features, ((pad_width, pad_width), (0, 0)), mode='edge')
        
        # Stack frames
        time_steps, freq_bins = features.shape
        stacked_features = np.zeros((time_steps, freq_bins * stack_size))
        
        for i in range(time_steps):
            start_idx = i
            end_idx = i + stack_size
            stacked_frame = padded_features[start_idx:end_idx].T.flatten()  # [freq * stack_size]
            stacked_features[i] = stacked_frame
            
        return stacked_features

    def wav2features(self, y, apply_stacking=True):
        """
        VoiceFilter-Lite 2020: Complete feature extraction pipeline
        Input: audio waveform
        Output: stacked filterbank features ready for VoiceFilter-Lite
        """
        # Extract log filterbank energies
        filterbank_features = self.wav2filterbank(y)
        
        # Apply frame stacking if requested
        if apply_stacking:
            features = self.stack_features(filterbank_features)
        else:
            features = filterbank_features
            
        return features

    # Legacy functions kept for compatibility (but not used in VoiceFilter-Lite 2020)
    def wav2spec(self, y):
        """Legacy function for 2019 compatibility - DEPRECATED in VoiceFilter-Lite"""
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - self.hp.audio.ref_level_db
        S, D = self.normalize(S), np.angle(D)
        S, D = S.T, D.T # to make [time, freq]
        return S, D

    def spec2wav(self, spectrogram, phase):
        """Legacy function for 2019 compatibility - NOT USED in VoiceFilter-Lite"""
        spectrogram, phase = spectrogram.T, phase.T
        # used during inference only
        # spectrogram: enhanced output
        # phase: use noisy input's phase, so no GLA is required
        S = self.db_to_amp(self.denormalize(spectrogram) + self.hp.audio.ref_level_db)
        return self.istft(S, phase)

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.hp.audio.n_fft,
                            hop_length=self.hp.audio.hop_length,
                            win_length=self.hp.audio.win_length)

    def istft(self, mag, phase):
        stft_matrix = mag * np.exp(1j*phase)
        return librosa.istft(stft_matrix,
                             hop_length=self.hp.audio.hop_length,
                             win_length=self.hp.audio.win_length)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def normalize(self, S):
        return np.clip(S / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db
