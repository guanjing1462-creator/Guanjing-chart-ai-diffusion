#!/usr/bin/env python3
"""音频特征提取：给有音频的谱面配 mel 频谱，并对齐到 46.44ms 谱面帧。

    python3 tools/build-audio-features.py --charts out/charts --pool /tmp/osz_x \
        --tensors out/mug_tensors --out out/audio_features

产出（每张有音频的谱一个 npz）：
    mel:   [128, 4096]  log-mel @46.44ms 谱面帧分辨率（8× 降采样自 5.805ms）
    index: out/audio_features/_index.jsonl  {id, set_dir, n_notes, ett}

匹配方式：读 .osu 的 AudioFilename，在 /tmp/osz_x/<set>/ 下找同名文件
（大小写不敏感）。重复利用 Mug 参数：sr 22050 / n_fft 512 / n_mels 128 / hop=128。
"""
import argparse
import glob
import json
import os

import numpy as np

SR = 22050
N_FFT = 512
HOP = N_FFT // 4          # 128 样本 = 5.805ms
N_MELS = 128
FRAME_MS = 2048 / SR / 2 * 1000   # 46.4399ms 谱面帧
MAX_NOTE_FRAME = 4096


AUDIO_INDEX = None


def build_audio_index(pool):
    """启动时全池 walk 一次建索引（避免每任务重复 os.walk）。"""
    global AUDIO_INDEX
    idx = {}
    for root, _, files in os.walk(pool):
        for f in files:
            key = f.lower()
            if key not in idx:
                idx[key] = os.path.join(root, f)
    AUDIO_INDEX = idx
    print(f'音频索引: {len(idx)} 文件', flush=True)
    return idx


def init_worker(pool):
    build_audio_index(pool)


def find_audio(pool, audio_name, set_hint=None):
    """按文件名查索引（O(1)）。"""
    if AUDIO_INDEX is None:
        build_audio_index(pool)
    return AUDIO_INDEX.get(audio_name.lower())


def load_mel(path, max_note_frame=MAX_NOTE_FRAME):
    """解码音频 → log-mel [128, T_audio] → 8× 池化到谱面帧并裁齐。"""
    import soundfile as sf
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    if sr != SR:
        import librosa
        data = librosa.resample(data[:, 0], orig_sr=sr, target_sr=SR)
    else:
        data = data[:, 0]
    import librosa
    mel = librosa.feature.melspectrogram(
        y=data.astype(np.float32), sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    logmel = np.log1p(mel).astype(np.float32)          # [128, T]
    # 8× 平均池化到 46.44ms 帧
    T = (logmel.shape[1] // 8) * 8
    mel_n = logmel[:, :T].reshape(N_MELS, T // 8, 8).mean(axis=2)
    if mel_n.shape[1] < max_note_frame:
        pad = np.zeros((N_MELS, max_note_frame - mel_n.shape[1]), dtype=np.float32)
        mel_n = np.concatenate([mel_n, pad], axis=1)
    return mel_n[:, :max_note_frame]


def read_audio_name(osu_path):
    sec = None
    for line in open(osu_path, encoding='utf-8', errors='ignore'):
        line = line.strip()
        if line == '[General]':
            sec = 'g'
            continue
        if line.startswith('['):
            sec = None
        if sec == 'g' and line.startswith('AudioFilename'):
            return line.split(':', 1)[1].strip()
    return None


def process_one(task):
    """worker：处理单张谱的音频 mel。返回 (cid, ok)。"""
    f, pool, out_dir = task
    cid = os.path.splitext(os.path.basename(f))[0]
    outp = os.path.join(out_dir, f'{cid}.npz')
    try:
        audio_name = read_audio_name(f)
        if not audio_name:
            return cid, False
        apath = find_audio(pool, audio_name)
        if not apath:
            return cid, False
        mel = load_mel(apath)
        T = (mel.shape[1] // 8) * 8
        mel_n = mel[:, :T].reshape(mel.shape[0], T // 8, 8).mean(axis=2)
        target = 4096
        if mel_n.shape[1] < target:
            mel_n = np.concatenate([mel_n, np.zeros((mel.shape[0], target - mel_n.shape[1]), dtype=np.float32)], axis=1)
        mel_n = mel_n[:, :target]
        np.savez_compressed(outp, mel=mel_n.astype(np.float16))
        return cid, True
    except Exception:
        return cid, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--charts', default='out/charts')
    ap.add_argument('--pool', default='/tmp/osz_x')
    ap.add_argument('--tensors', default='out/mug_tensors')
    ap.add_argument('--out', default='out/audio_features')
    ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3,
                    help='并行 worker 数；1 = 串行（Pool 死锁时用）')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, '_index.jsonl')
    done = set()
    if os.path.exists(index_path):
        for l in open(index_path):
            try:
                done.add(json.loads(l)['id'])
            except Exception:
                pass

    made = skip = no_audio = 0
    idx_f = open(index_path, 'a')
    files = sorted(glob.glob(os.path.join(args.charts, '*.osu')))
    tasks = []
    for f in files:
        cid = os.path.splitext(os.path.basename(f))[0]
        if cid in done:
            skip += 1
            continue
        if os.path.exists(os.path.join(args.out, f'{cid}.npz')):
            skip += 1
            continue
        tasks.append((f, args.pool, args.out))
    print(f'待处理 {len(tasks)}（跳过 {skip}）', flush=True)
    from concurrent.futures import ProcessPoolExecutor
    n_ok = 0
    if args.workers <= 1:
        build_audio_index(args.pool)
        for task in tasks:
            cid, ok = process_one(task)
            if ok:
                idx_f.write(json.dumps({'id': cid}) + '\n')
                n_ok += 1
                made += 1
            else:
                no_audio += 1
            if made % 100 == 0 and made:
                idx_f.flush()
                print(f'进度 {made}  ok率 {n_ok}/{made}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=init_worker,
                                 initargs=(args.pool,)) as ex:
            for cid, ok in ex.map(process_one, tasks, chunksize=4):
                if ok:
                    idx_f.write(json.dumps({'id': cid}) + '\n')
                    n_ok += 1
                    made += 1
                else:
                    no_audio += 1
                if made % 100 == 0 and made:
                    idx_f.flush()
                    print(f'进度 {made}  ok率 {n_ok}/{made}', flush=True)
    idx_f.close()
    print(f'完成：本批 +{made}（音频命中 {n_ok}）', flush=True)


if __name__ == '__main__':
    main()
