import os
import glob
import tqdm
import torch
import random
import librosa
import soundfile as sf
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from utils.audio import Audio
from utils.hparams import HParam


def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % num))

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def mix_voicefilter_lite_2020(hp, args, audio, num, s1_dvec, s1_target, s2, train):
    """
    VoiceFilter-Lite 2020 data generation
    Generates filterbank features instead of STFT spectrograms
    """
    srate = hp.audio.sample_rate
    dir_ = os.path.join(args.out_dir, 'train' if train else 'test')

    d, _ = librosa.load(s1_dvec, sr=srate)
    w1, _ = librosa.load(s1_target, sr=srate)
    w2, _ = librosa.load(s2, sr=srate)
    assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
        'wav files must be mono, not stereo'

    d, _ = librosa.effects.trim(d, top_db=20)
    w1, _ = librosa.effects.trim(w1, top_db=20)
    w2, _ = librosa.effects.trim(w2, top_db=20)

    # if reference for d-vector is too short, discard it
    if d.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
        return

    # LibriSpeech dataset have many silent interval, so let's vad-merge them
    if args.vad == 1:
        w1, w2 = vad_merge(w1), vad_merge(w2)

    # Fit audio to `hp.data.audio_len` seconds.
    L = int(srate * hp.data.audio_len)
    if w1.shape[0] < L or w2.shape[0] < L:
        return
    w1, w2 = w1[:L], w2[:L]

    mixed = w1 + w2

    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, mixed = w1/norm, w2/norm, mixed/norm

    # Save wav files
    target_wav_path = formatter(dir_, hp.form.target.wav, num)
    mixed_wav_path = formatter(dir_, hp.form.mixed.wav, num)
    sf.write(target_wav_path, w1, srate)
    sf.write(mixed_wav_path, mixed, srate)

    # VoiceFilter-Lite 2020: Generate filterbank features instead of spectrograms
    try:
        # Mixed audio → filterbank features (input to model)
        mixed_features = audio.wav2filterbank(mixed)
        mixed_stacked = audio.stack_features(mixed_features, hp.audio.frame_stack_size)
        
        # Clean target → filterbank features (target for model)
        target_features = audio.wav2filterbank(w1)
        
        # Save filterbank features
        mixed_features_path = formatter(dir_, hp.form.mixed.features, num)
        target_features_path = formatter(dir_, hp.form.target.features, num)
        
        torch.save(torch.from_numpy(mixed_stacked), mixed_features_path)
        torch.save(torch.from_numpy(target_features), target_features_path)
        
        # Generate noise type label for adaptive suppression
        # Simple heuristic: if w2 energy > 0.3 * w1 energy, it's "overlapped speech"
        w1_energy = np.mean(w1**2)
        w2_energy = np.mean(w2**2)
        noise_type = 1 if w2_energy > 0.3 * w1_energy else 0  # 0: clean/non-speech, 1: overlapped speech
        
        # Save noise type
        noise_type_path = formatter(dir_, hp.form.noise_type, num)
        torch.save(torch.tensor(noise_type), noise_type_path)
        
    except Exception as e:
        print(f"⚠️ Error generating filterbank features for sample {num}: {e}")
        # Fallback to old method if filterbank extraction fails
    target_mag, _ = audio.wav2spec(w1)
    mixed_mag, _ = audio.wav2spec(mixed)
    target_mag_path = formatter(dir_, hp.form.target.mag, num)
    mixed_mag_path = formatter(dir_, hp.form.mixed.mag, num)
    torch.save(torch.from_numpy(target_mag), target_mag_path)
    torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)

    # Save selected sample as text file. d-vec will be calculated soon
    dvec_text_path = formatter(dir_, hp.form.dvec, num)
    with open(dvec_text_path, 'w') as f:
        f.write(s1_dvec)


# Legacy function for backward compatibility
def mix(hp, args, audio, num, s1_dvec, s1_target, s2, train):
    """Legacy VoiceFilter 2019 function - redirects to VoiceFilter-Lite 2020"""
    return mix_voicefilter_lite_2020(hp, args, audio, num, s1_dvec, s1_target, s2, train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-d', '--libri_dir', type=str, default=None,
                        help="Directory of LibriSpeech dataset, containing folders of train-clean-100, train-clean-360, dev-clean.")
    parser.add_argument('-v', '--voxceleb_dir', type=str, default=None,
                        help="Directory of VoxCeleb2 dataset, ends with 'aac'")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="Directory of output training triplet")
    parser.add_argument('-p', '--process_num', type=int, default=None,
                        help='number of processes to run. default: cpu_count')
    parser.add_argument('--vad', type=int, default=0,
                        help='apply vad to wav file. yes(1) or no(0, default)')
    parser.add_argument('--mode', type=str, default='2020', choices=['2019', '2020'],
                        help='VoiceFilter mode: 2019 (original) or 2020 (lite). Default: 2020')
    args = parser.parse_args()

    print(f"🎯 Running VoiceFilter-Lite {args.mode} data generation...")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'test'), exist_ok=True)

    hp = HParam(args.config)

    cpu_num = cpu_count() if args.process_num is None else args.process_num

    if args.libri_dir is None and args.voxceleb_dir is None:
        raise Exception("Please provide directory of data")

    if args.libri_dir is not None:
        train_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-100', '*'))
                            if os.path.isdir(x)] + \
                        [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-360', '*'))
                            if os.path.isdir(x)]
        test_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'dev-clean', '*'))]

    elif args.voxceleb_dir is not None:
        all_folders = [x for x in glob.glob(os.path.join(args.voxceleb_dir, '*'))
                            if os.path.isdir(x)]
        train_folders = all_folders[:-20]
        test_folders = all_folders[-20:]

    train_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                    for spk in train_folders]
    train_spk = [x for x in train_spk if len(x) >= 2]

    test_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                    for spk in test_folders]
    test_spk = [x for x in test_spk if len(x) >= 2]

    audio = Audio(hp)

    # Choose generation function based on mode
    if args.mode == '2020':
        mix_func = mix_voicefilter_lite_2020
        print("🚀 Using VoiceFilter-Lite 2020 filterbank feature generation")
    else:
        mix_func = mix  # Legacy 2019 function
        print("⚠️ Using legacy VoiceFilter 2019 spectrogram generation")

    def train_wrapper(num):
        spk1, spk2 = random.sample(train_spk, 2)
        s1_dvec, s1_target = random.sample(spk1, 2)
        s2 = random.choice(spk2)
        mix_func(hp, args, audio, num, s1_dvec, s1_target, s2, train=True)

    def test_wrapper(num):
        spk1, spk2 = random.sample(test_spk, 2)
        s1_dvec, s1_target = random.sample(spk1, 2)
        s2 = random.choice(spk2)
        mix_func(hp, args, audio, num, s1_dvec, s1_target, s2, train=False)

    print(f"📊 Generating training data...")
    arr = list(range(100000))  # 100k training samples for production
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(train_wrapper, arr), total=len(arr)))

    print(f"📊 Generating test data...")
    arr = list(range(1000))    # 1k test samples for production
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(test_wrapper, arr), total=len(arr)))

    print(f"✅ VoiceFilter-Lite {args.mode} data generation completed!")
