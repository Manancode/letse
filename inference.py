import os
import glob
import torch
import librosa
import argparse
import numpy as np

from utils.audio import Audio
from utils.hparams import HParam
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder


def main(args, hp):
    with torch.no_grad():
        # VoiceFilter-Lite 2020 Model
        model = VoiceFilter(hp).cuda()
        chkpt_model = torch.load(args.checkpoint_path)['model']
        model.load_state_dict(chkpt_model)
        model.eval()

        # Speaker embedder (unchanged from 2019)
        embedder = SpeechEmbedder(hp).cuda()
        chkpt_embed = torch.load(args.embedder_path)
        embedder.load_state_dict(chkpt_embed)
        embedder.eval()

        # Audio processor (VoiceFilter-Lite 2020)
        audio = Audio(hp)
        
        # Extract speaker embedding (d-vector) from reference audio
        dvec_wav, _ = librosa.load(args.reference_file, sr=hp.audio.sample_rate)
        dvec_mel = audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float().cuda()
        dvec = embedder(dvec_mel)
        dvec = dvec.unsqueeze(0)

        # Load and process mixed audio
        mixed_wav, _ = librosa.load(args.mixed_file, sr=hp.audio.sample_rate)
        
        # VoiceFilter-Lite 2020: Extract filterbank features instead of spectrograms
        mixed_features = audio.wav2features(mixed_wav)
        mixed_features = torch.from_numpy(mixed_features).float().cuda()
        mixed_features = mixed_features.unsqueeze(0)  # Add batch dimension

        print(f"Input features shape: {mixed_features.shape}")
        print(f"Speaker embedding shape: {dvec.shape}")

        # VoiceFilter-Lite 2020: Get enhanced features with adaptive suppression
        if args.disable_adaptive_suppression:
            print("Adaptive suppression disabled")
            enhanced_features, noise_pred = model.separate_speech(
                mixed_features, dvec, apply_adaptive_suppression=False)
        else:
            print("Using adaptive suppression")
            enhanced_features, noise_pred = model.separate_speech(
                mixed_features, dvec, apply_adaptive_suppression=True)

        print(f"Enhanced features shape: {enhanced_features.shape}")
        print(f"Noise prediction shape: {noise_pred.shape}")

        # Convert enhanced features back to numpy
        enhanced_features = enhanced_features[0].cpu().detach().numpy()  # Remove batch dim
        noise_pred = noise_pred[0].cpu().detach().numpy()

        # Print noise type analysis
        speech_prob = noise_pred[:, 1].mean()
        non_speech_prob = noise_pred[:, 0].mean()
        print(f"Average speech noise probability: {speech_prob:.3f}")
        print(f"Average non-speech noise probability: {non_speech_prob:.3f}")

        # VoiceFilter-Lite 2020: Enhanced features are ready for ASR
        # For demo purposes, we can convert back to audio for listening
        if args.output_audio:
            try:
                # Convert enhanced features back to audio (approximation for demo)
                enhanced_audio = features_to_audio_approximation(enhanced_features, audio, hp)
                
                # Save enhanced audio
                os.makedirs(args.out_dir, exist_ok=True)
                out_path = os.path.join(args.out_dir, 'enhanced_audio.wav')
                librosa.output.write_wav(out_path, enhanced_audio, sr=hp.audio.sample_rate)
                print(f"Enhanced audio saved to: {out_path}")
                
            except Exception as e:
                print(f"Warning: Audio generation failed: {e}")
        
        # Save enhanced features (main VoiceFilter-Lite 2020 output)
        os.makedirs(args.out_dir, exist_ok=True)
        features_path = os.path.join(args.out_dir, 'enhanced_features.npy')
        np.save(features_path, enhanced_features)
        print(f"Enhanced features saved to: {features_path}")
        
        # Save noise predictions
        noise_pred_path = os.path.join(args.out_dir, 'noise_predictions.npy')
        np.save(noise_pred_path, noise_pred)
        print(f"Noise predictions saved to: {noise_pred_path}")


def features_to_audio_approximation(enhanced_features, audio, hp):
    """
    Convert enhanced features back to audio for demo purposes
    This is NOT part of the VoiceFilter-Lite 2020 pipeline but useful for evaluation
    
    Note: This is a rough approximation since VoiceFilter-Lite works in feature domain
    """
    try:
        # Extract base filterbank features (remove stacking)
        time_steps, stacked_dim = enhanced_features.shape
        stack_size = hp.audio.frame_stack_size  # 3
        base_dim = hp.audio.num_filterbanks  # 128
        
        # Reshape and extract middle frame
        features_reshaped = enhanced_features.reshape(time_steps, stack_size, base_dim)
        middle_frame_idx = stack_size // 2
        base_features = features_reshaped[:, middle_frame_idx, :]  # [T, 128]
        
        # Convert log filterbank back to magnitude (rough approximation)
        magnitude_approx = np.exp(base_features)  # [T, 128]
        
        # Pad or truncate to match STFT dimensions if needed
        target_freq_bins = hp.audio.num_freq  # 601
        if magnitude_approx.shape[1] < target_freq_bins:
            # Pad with zeros
            pad_width = target_freq_bins - magnitude_approx.shape[1]
            magnitude_approx = np.pad(magnitude_approx, ((0, 0), (0, pad_width)), mode='constant')
        elif magnitude_approx.shape[1] > target_freq_bins:
            # Truncate
            magnitude_approx = magnitude_approx[:, :target_freq_bins]
        
        # Transpose for librosa format [F, T]
        magnitude_approx = magnitude_approx.T
        
        # Simple audio synthesis using Griffin-Lim
        audio_approx = librosa.griffinlim(magnitude_approx, 
                                        hop_length=hp.audio.hop_length,
                                        win_length=hp.audio.win_length,
                                        n_iter=32)
        
        return audio_approx
        
    except Exception as e:
        print(f"Warning: Features to audio conversion failed: {e}")
        # Return silence if conversion fails
        return np.zeros(48000)  # 3 seconds of silence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VoiceFilter-Lite 2020 Inference')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-e', '--embedder_path', type=str, required=True,
                        help="path of embedder model pt file")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="path of VoiceFilter-Lite checkpoint pt file")
    parser.add_argument('-m', '--mixed_file', type=str, required=True,
                        help='path of mixed wav file')
    parser.add_argument('-r', '--reference_file', type=str, required=True,
                        help='path of reference wav file for speaker enrollment')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='directory of output')
    parser.add_argument('--disable_adaptive_suppression', action='store_true',
                        help='disable adaptive suppression (use basic masking only)')
    parser.add_argument('--output_audio', action='store_true',
                        help='generate audio output for demo (approximation)')

    args = parser.parse_args()

    hp = HParam(args.config)
    
    print("VoiceFilter-Lite 2020 Inference")
    print("=" * 40)
    print(f"Config: {args.config}")
    print(f"Mixed file: {args.mixed_file}")
    print(f"Reference file: {args.reference_file}")
    print(f"Output directory: {args.out_dir}")
    print(f"Adaptive suppression: {'Disabled' if args.disable_adaptive_suppression else 'Enabled'}")
    print("=" * 40)

    main(args, hp)
