#!/usr/bin/env python3
"""为有音频的歌生成「对齐 1/16 网格」的音频条件特征（扩散模型 audio-condition 分支用）。

Mug-Diffusion 用 mel 频谱当条件；这里做同构的小样本版本：每首曲子按分析文件的
offsetMs / stepMs 切格，每格给 4 个 onset 强度（0–1）。

每格的 4 个值：
    ch0  低频 onset 强度    0–200 Hz
    ch1  中频 onset 强度    200–2000 Hz
    ch2  高频 onset 强度    2000–8000 Hz
    ch3  总   onset 强度    0–8000 Hz

算法（单声道 22.05kHz，n_fft=512 / hop=128 / 128 带 mel，与 Mug 的 mel 参数一致）：
    |STFT| → mel(0–8kHz) → log1p → 沿时间正向差分取正（谱通量）→ 按带求和
    → 每格取格内最大值（保住瞬态）→ 每首每通道按 P99 归一到 0–1
    帧→格时刻用「帧中心」((c+1)*hop + n_fft/2)，合成脉冲轨验证 100% 落在正确格

输出：
    out/multisong/<分析文件名>-cond.npz   键 feat [T,4] float16
    out/multisong/cond-meta.json          每首网格参数、归一化系数、与 fx.onset 的相关性

⚠ 只有 6 首歌有音频，这份条件数据是小样本，用于验证"音频条件分支"的接口，
   不是全量训练数据（全量训练仍要回到有音频的全量谱面集）。

一次只处理一首歌；STFT 分块算，内存峰值与音频长度无关（约几 MB）。
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

# 与 Mug-Diffusion / 本仓 build-audio-features.py 的 mel 参数对齐（512/128/128 @22.05kHz），
# 这样以后把 4 维带能量换成完整 mel 张量时，网格与帧率都不用改。
SR = 22050
N_FFT = 512
HOP = 128                 # 5.805ms
N_MELS = 128
FMIN, FMAX = 0.0, 8000.0
BANDS = ((0.0, 200.0), (200.0, 2000.0), (2000.0, 8000.0))
CH_NAMES = ['low_0_200', 'mid_200_2000', 'high_2000_8000', 'onset_total']
BLOCK = 4096              # 每次 STFT 处理的帧数（≈23.8s），控制内存峰值


def onset_bands(y, sr=SR):
    """分块算 4 通道 onset 强度包络，返回 [4, F] float32（F ≈ len(y)/HOP）。"""
    import librosa

    n_frames = 1 + (len(y) - N_FFT) // HOP
    if n_frames < 3:
        return np.zeros((4, 0), dtype=np.float32)

    mel_fb = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    mel_hz = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    masks = [mel_hz < BANDS[0][1],
             (mel_hz >= BANDS[1][0]) & (mel_hz < BANDS[1][1]),
             mel_hz >= BANDS[2][0]]

    out = np.zeros((4, n_frames - 1), dtype=np.float32)
    prev = None
    for i0 in range(0, n_frames, BLOCK):
        i1 = min(n_frames, i0 + BLOCK)
        seg = y[i0 * HOP:(i1 - 1) * HOP + N_FFT]
        S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP, center=False))
        L = np.log1p(mel_fb @ (S ** 2))          # [n_mels, i1-i0]
        if i0 == 0:
            flux = np.diff(L, axis=1)
            col0 = 0
        else:
            flux = np.diff(np.concatenate([prev[:, None], L], axis=1), axis=1)
            col0 = i0 - 1
        flux = np.maximum(flux, 0.0)             # 只保留上升沿
        prev = L[:, -1].copy()
        m = flux.shape[1]
        out[0, col0:col0 + m] = flux[masks[0]].sum(axis=0)
        out[1, col0:col0 + m] = flux[masks[1]].sum(axis=0)
        out[2, col0:col0 + m] = flux[masks[2]].sum(axis=0)
        out[3, col0:col0 + m] = flux.sum(axis=0)
        del S, L, flux
    return out


def grid_cells(env, offset_ms, step_ms, steps, sr=SR):
    """把帧级包络聚到网格：每格取 [t0, t0+step) 帧上的最大值。返回 [T,4] float32。

    第 c 列通量对应 STFT 帧 c+1，该帧中心在 ((c+1)*HOP + N_FFT/2) 采样处。
    """
    feat = np.zeros((steps, 4), dtype=np.float32)
    if env.shape[1] == 0:
        return feat, 0
    t_frame = ((np.arange(env.shape[1]) + 1) * HOP + N_FFT // 2) / sr    # 秒（帧中心）
    t0 = (offset_ms + np.arange(steps, dtype=np.float64) * step_ms) / 1000.0
    t1 = t0 + step_ms / 1000.0
    i0 = np.searchsorted(t_frame, t0, side='left')
    i1 = np.searchsorted(t_frame, t1, side='left')
    pad = 0
    for k in range(steps):
        a, b = i0[k], i1[k]
        if a >= env.shape[1]:
            pad += 1
            continue
        if b <= a:
            b = a + 1
        feat[k] = env[:, a:min(b, env.shape[1])].max(axis=1)
    return feat, pad


def normalize_p99(x):
    """每通道按 P99 归一到 0–1，返回 (归一化结果, 每通道系数)。"""
    scale = np.zeros(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        p = float(np.percentile(x[:, j], 99)) if x.shape[0] else 0.0
        scale[j] = p if p > 1e-9 else 1.0
    return np.clip(x / scale, 0.0, 1.0), scale


def grid_sanity(y, offset_ms, step_ms, sr=SR):
    """网格可信度：检测到的真实 onset 有多少落在网格线上。

    返回 (对齐比例, 半格平移后的对照比例)。随机时刻的理论基线 ≈ 0.30（±15% 容差）。
    """
    import librosa

    try:
        t = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units='time', backtrack=False)
    except Exception:
        return 0.0, 0.0
    if len(t) < 32:
        return 0.0, 0.0
    step = step_ms / 1000.0

    def frac(shift_ms):
        ph = ((t * 1000.0 - (offset_ms + shift_ms)) % step_ms) / step_ms
        return float(np.mean((ph < 0.15) | (ph > 0.85)))

    return frac(0.0), frac(step_ms / 2.0)


def corr_with_fx(feat_col, fx_onset):
    """与 fx.json 里的 onset 包络对齐相关性（自动容忍 ±3 格偏移）。"""
    a_raw = np.asarray(fx_onset, dtype=np.float64)
    best = (0.0, 0)
    for shift in range(-3, 4):
        if shift >= 0:
            a, b = a_raw[shift:], feat_col[:len(a_raw) - shift]
        else:
            a, b = a_raw[:len(a_raw) + shift], feat_col[-shift:]
        n = min(len(a), len(b))
        if n < 16:
            continue
        a, b = a[:n], b[:n]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best[0]:
            best = (r, shift)
    return best


def process_song(song_id, meta_path, out_dir, sr=SR, verbose=True):
    import librosa

    t_start = time.time()
    meta = json.load(open(meta_path, encoding='utf-8'))
    src = meta.get('source', '')
    audio_path = src if os.path.isabs(src) else src
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'音频不存在：{audio_path}（来自 {meta_path}）')

    offset_ms = float(meta['offsetMs'])
    step_ms = float(meta['stepMs'])
    steps = int(meta['steps'])

    t0 = time.time()
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    load_s = time.time() - t0
    dur = len(y) / sr

    env = onset_bands(y, sr)
    align_frac, align_ctrl = grid_sanity(y, offset_ms, step_ms, sr)
    del y
    feat_raw, pad = grid_cells(env, offset_ms, step_ms, steps)
    del env
    feat, scale = normalize_p99(feat_raw)

    r, shift = corr_with_fx(feat[:, 3], meta.get('onset', []))
    feat16 = feat.astype(np.float16)
    out_path = os.path.join(out_dir, f'{song_id}-cond.npz')
    np.savez_compressed(out_path, feat=feat16)

    info = {
        'song_id': song_id,
        'audio': audio_path,
        'analysis': meta_path,
        'out': out_path,
        'sample_rate': sr,
        'duration_sec': round(dur, 3),
        'offset_ms': round(offset_ms, 3),
        'step_ms': round(step_ms, 4),
        'steps': steps,
        'feat_shape': list(feat16.shape),
        'dtype': 'float16',
        'channels': CH_NAMES,
        'p99_scale': [round(float(s), 6) for s in scale],
        'padded_cells': int(pad),
        'mean_per_channel': [round(float(x), 4) for x in feat16.mean(axis=0)],
        'ones_per_channel': [round(float(x), 4) for x in (feat16 > 0.5).mean(axis=0)],
        'corr_with_fx_onset': round(r, 4),
        'corr_best_shift_cells': int(shift),
        'grid_onset_align': round(align_frac, 4),
        'grid_onset_align_halfcell_control': round(align_ctrl, 4),
        'seconds': round(time.time() - t_start, 1),
        'load_seconds': round(load_s, 1),
    }
    if verbose:
        print(f'  {song_id}: T={steps}  {dur:.1f}s  offset={offset_ms:.1f}ms  step={step_ms:.2f}ms')
        print(f'    归一化 P99={np.round(scale, 3).tolist()}  补零格={pad}  '
              f'与 fx.onset 相关 r={r:.3f}(shift {shift})  {info["seconds"]}s')
        print(f'    网格自检：真实 onset 落网格线 {align_frac:.2f}（半格平移对照 {align_ctrl:.2f}，随机基线 0.30）')
        print(f'    → {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)')
    return info


def discover(out_dir='out/multisong'):
    """找所有 <歌名>-fx.json（排除 -det / -raw 等衍生文件）。"""
    return sorted(glob.glob(os.path.join(out_dir, '*-fx.json')))


def main():
    ap = argparse.ArgumentParser(description='生成对齐 1/16 网格的音频条件特征')
    ap.add_argument('--out-dir', default='out/multisong')
    ap.add_argument('--song', default='', help='只处理名字含该串的歌（默认全部）')
    ap.add_argument('--force', action='store_true', help='覆盖已有输出')
    ap.add_argument('--sr', type=int, default=SR)
    a = ap.parse_args()

    files = discover(a.out_dir)
    if a.song:
        files = [f for f in files if a.song in os.path.basename(f)]
    if not files:
        print('没有找到分析文件 *-fx.json', file=sys.stderr)
        return 1

    print(f'待处理 {len(files)} 首（一次一首，逐首释放内存）')
    all_info = []
    for f in files:
        song_id = os.path.basename(f)[:-len('-fx.json')]
        dst = os.path.join(a.out_dir, f'{song_id}-cond.npz')
        if os.path.exists(dst) and not a.force:
            print(f'  {song_id}: 已存在，跳过（--force 覆盖）')
            continue
        try:
            all_info.append(process_song(song_id, f, a.out_dir, sr=a.sr))
        except Exception as e:                     # 单首失败不影响其它
            print(f'  {song_id}: 失败 {type(e).__name__}: {e}', file=sys.stderr)

    # 合并写 meta（保留旧条目，避免 --song 单跑时丢信息）
    meta_path = os.path.join(a.out_dir, 'cond-meta.json')
    old = {}
    if os.path.exists(meta_path):
        try:
            old = json.load(open(meta_path, encoding='utf-8')).get('songs', {})
        except Exception:
            old = {}
    songs = dict(old)
    for info in all_info:
        songs[info['song_id']] = info
    payload = {
        'note': '小样本：仅 6 首有音频；用于音频条件分支的接口验证，不是全量训练数据',
        'features': {
            'channels': CH_NAMES,
            'unit': '归一化 onset 强度 0–1（每首每通道 P99 缩放）',
            'frame': f'STFT n_fft={N_FFT} hop={HOP} → mel {N_MELS} 带 0–8kHz → log1p → 正差分 → 带内求和 → 格内最大',
            'grid': '以分析文件 offsetMs 为起点、stepMs 为步长（1/16 拍）',
        },
        'songs': songs,
    }
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f'\n元数据：{meta_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
