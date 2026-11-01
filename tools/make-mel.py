#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make-mel: 任意音频 → 扩散模型要的 mel [128,4096]（严格照 FEATURES.md 口径）。

sr 22050 / n_fft 512 / hop 128（5.805ms）/ n_mels 128 / log1p / 8× 平均池化到
46.44ms 帧 / 不足 4096 帧补零。参数错了模型能跑但条件会错位。

用法：
  python3 tools/make-mel.py --audio "ref/INTERNET YAMERO.ogg" -o out/audio_features/INTERNET_YAMERO.npz
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('-o', required=True)
    ap.add_argument('--max-frames', type=int, default=4096)
    args = ap.parse_args()
    import librosa
    import soundfile as sf
    SR, N_FFT, HOP, N_MELS = 22050, 512, 128, 128
    data, sr = sf.read(args.audio, dtype='float32', always_2d=True)
    mono = data[:, 0]
    if sr != SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)
    mel = librosa.feature.melspectrogram(y=mono.astype(np.float32), sr=SR,
                                         n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    logmel = np.log1p(mel).astype(np.float32)
    T = (logmel.shape[1] // 8) * 8
    mel_n = logmel[:, :T].reshape(N_MELS, T // 8, 8).mean(axis=2)
    if mel_n.shape[1] < args.max_frames:
        mel_n = np.concatenate(
            [mel_n, np.zeros((N_MELS, args.max_frames - mel_n.shape[1]), np.float32)], axis=1)
    mel_n = mel_n[:, :args.max_frames].astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.o)), exist_ok=True)
    np.savez_compressed(args.o, mel=mel_n)
    print(f'{args.o}  mel {mel_n.shape}  {mel_n.shape[1]*46.4399/1000:.1f}s')


if __name__ == '__main__':
    main()
