import os
import glob
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.audio import Audio


def create_dataloader(hp, args, train):
    def train_collate_fn(batch):
        dvec_list = list()
        target_features_list = list()
        mixed_features_list = list()

        for dvec_mel, target_features, mixed_features in batch:
            dvec_list.append(dvec_mel)
            target_features_list.append(target_features)
            mixed_features_list.append(mixed_features)
        target_features_list = torch.stack(target_features_list, dim=0)
        mixed_features_list = torch.stack(mixed_features_list, dim=0)

        return dvec_list, target_features_list, mixed_features_list

    def test_collate_fn(batch):
        return batch

    if train:
        return DataLoader(dataset=VFDataset(hp, args, True),
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    else:
        return DataLoader(dataset=VFDataset(hp, args, False),
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)


class VFDataset(Dataset):
    def __init__(self, hp, args, train):
        def find_all(file_format):
            return sorted(glob.glob(os.path.join(self.data_dir, file_format)))
            
        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = hp.data.train_dir if train else hp.data.test_dir

        self.dvec_list = find_all(hp.form.dvec)
        self.target_wav_list = find_all(hp.form.target.wav)
        self.mixed_wav_list = find_all(hp.form.mixed.wav)
        
        # VoiceFilter-Lite 2020: Use filterbank features
        self.target_features_list = find_all(hp.form.target.features)
        self.mixed_features_list = find_all(hp.form.mixed.features)
        
        # Legacy 2019 format for compatibility
        self.target_mag_list = find_all(hp.form.target.mag)
        self.mixed_mag_list = find_all(hp.form.mixed.mag)

        # Check if VoiceFilter-Lite 2020 features exist, otherwise fall back to 2019
        self.use_features_2020 = (len(self.target_features_list) > 0 and 
                                  len(self.mixed_features_list) > 0)
        
        if self.use_features_2020:
            print("Using VoiceFilter-Lite 2020 filterbank features")
            assert len(self.dvec_list) == len(self.target_wav_list) == len(self.mixed_wav_list) == \
                len(self.target_features_list) == len(self.mixed_features_list), \
                "number of training files must match"
        else:
            print("WARNING: VoiceFilter-Lite 2020 features not found, using legacy 2019 format")
        assert len(self.dvec_list) == len(self.target_wav_list) == len(self.mixed_wav_list) == \
                len(self.target_mag_list) == len(self.mixed_mag_list), \
                "number of training files must match"
        
        assert len(self.dvec_list) != 0, "no training file found"

        self.audio = Audio(hp)

    def __len__(self):
        return len(self.dvec_list)

    def __getitem__(self, idx):
        with open(self.dvec_list[idx], 'r') as f:
            dvec_path = f.readline().strip()

        dvec_wav, _ = librosa.load(dvec_path, sr=self.hp.audio.sample_rate)
        dvec_mel = self.audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float()

        if self.train: # need to be fast
            if self.use_features_2020:
                # VoiceFilter-Lite 2020: Load preprocessed filterbank features
                target_features = torch.load(self.target_features_list[idx]).float()
                mixed_features = torch.load(self.mixed_features_list[idx]).float()
                return dvec_mel, target_features, mixed_features
            else:
                # Legacy 2019: Load magnitude spectrograms and convert to features
            target_mag = torch.load(self.target_mag_list[idx])
            mixed_mag = torch.load(self.mixed_mag_list[idx])
                
                # Convert spectrograms to features on-the-fly (slower but works)
                target_features = self._spec_to_features(target_mag.numpy())
                mixed_features = self._spec_to_features(mixed_mag.numpy())
                
                target_features = torch.from_numpy(target_features).float()
                mixed_features = torch.from_numpy(mixed_features).float()
                
                return dvec_mel, target_features, mixed_features
        else:
            # Test mode: always compute features from audio
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr=self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr=self.hp.audio.sample_rate)
            
            # VoiceFilter-Lite 2020: Extract filterbank features
            target_features = self.audio.wav2features(target_wav)
            mixed_features = self.audio.wav2features(mixed_wav)
            
            target_features = torch.from_numpy(target_features).float()
            mixed_features = torch.from_numpy(mixed_features).float()
            
            # For test mode, we might still need phase for audio reconstruction
            if self.use_features_2020:
                # No phase needed for VoiceFilter-Lite 2020 (direct ASR features)
                mixed_phase = None
            else:
                # Legacy: compute phase for potential audio reconstruction
                _, mixed_phase = self.wav2magphase(self.mixed_wav_list[idx])
            
            return dvec_mel, target_wav, mixed_wav, target_features, mixed_features, mixed_phase

    def _spec_to_features(self, magnitude_spec):
        """
        Convert magnitude spectrogram to filterbank features
        This is a fallback for when 2020 features are not pre-computed
        """
        # This is a simplified conversion - ideally features should be pre-computed
        # For now, just reshape and truncate/pad to match expected dimensions
        time_steps = magnitude_spec.shape[0]
        target_dim = self.hp.model.input_dim
        
        if magnitude_spec.shape[1] > target_dim:
            # Truncate frequency dimension
            features = magnitude_spec[:, :target_dim]
        else:
            # Pad frequency dimension
            pad_width = target_dim - magnitude_spec.shape[1]
            features = np.pad(magnitude_spec, ((0, 0), (0, pad_width)), mode='constant')
            
        return features

    def wav2magphase(self, path):
        """Legacy function for 2019 compatibility"""
        wav, _ = librosa.load(path, self.hp.audio.sample_rate)
        mag, phase = self.audio.wav2spec(wav)
        return mag, phase

    def wav2features(self, path):
        """
        VoiceFilter-Lite 2020: Extract filterbank features from audio file
        """
        wav, _ = librosa.load(path, self.hp.audio.sample_rate)
        features = self.audio.wav2features(wav)
        return features
