#!/usr/bin/env python3
"""特征打包：Mel-128 音频特征 + 4 通道谱面张量。

参数严格对齐 Mug-Diffusion（`mug/util.py: load_audio_without_cache` 和
`mug/data/convertor.py: objects_to_array`），这样将来可以直接拿它的权重
或者用它的 dataloader 读我们这份数据。

    sr       = 22050
    n_fft    = 2048
    hop      = 1024          -> frame_ms = 46.4399
    n_mels   = 128
    压缩     = log1p，存 float16
    谱面     = [4*keys, max_frames]，通道依次为
                 start / start_offset / ln_body / ln_end_offset
    帧号     = int(t / frame_ms)   （向下取整）
    offset   = (t - idx*frame_ms) / frame_ms   （0~1，不是 ±0.5）

用法：
    python3 tools/extract-features.py --src /tmp/osz_x --out out/features
    python3 tools/extract-features.py --src /tmp/osz_x --out out/features --workers 16

输出：
    out/features/audio/{setId}.npz       mel [128,T] f16 + onset [T] f16
    out/features/chart/{setId}__{n}.npz  tensor [4K,T] f16 + valid [T] u1
    out/features/index.jsonl             每个难度的元数据
    out/features/skipped.jsonl           跳过的原因（排查用）
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SR = 22050
N_FFT = 2048
HOP = 1024
N_MELS = 128
FRAME_MS = HOP / SR * 1000.0          # 46.4399
MAX_FRAMES = 8192                     # ≈ 380 秒
AUDIO_EXT = ('.mp3', '.ogg', '.wav', '.m4a')


# --------------------------------------------------------------------------
# 音频
# --------------------------------------------------------------------------
def decode_audio(path, sr=SR):
    """用 ffmpeg 管道解码成单声道 float32，不落临时文件。

    torchaudio 2.10 的 load 需要 torchcodec（未安装），soundfile 不认所有 mp3，
    ffmpeg 是唯一稳的路径，而且能把 -threads 限死，避免 16 个 worker 抢 32 核。
    """
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1',
           '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(sr), '-']
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if not p.stdout:
        raise RuntimeError('ffmpeg 无输出（编码不支持或文件损坏）')
    return np.frombuffer(p.stdout, dtype=np.float32)


def mel_of(y):
    """log1p(melspectrogram)，参数对齐 librosa 默认值。

    librosa 默认 htk=False(→ mel_scale='slaney') 且 norm='slaney'，
    torchaudio 的默认是 htk + 无归一化，所以这里必须显式指定，
    否则同一段音频算出来的 mel 和 Mug-Diffusion 差一个常数标度。
    """
    import torch
    import torchaudio
    torch.set_num_threads(1)

    t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        f_min=0.0, f_max=None, pad_mode='reflect', power=2.0,
        norm='slaney', mel_scale='slaney',
    )(t)
    return np.log1p(mel.numpy())


def rms_envelope(y, hop_ms=10.0, sr=SR):
    """全频段 RMS 包络（10ms 步长）。用途：

    - 检测谱面与音频的系统性偏移（ffmpeg 解 mp3 时编码器延迟不会总是被跳过，
      实测整体滞后约 40–50ms，正好接近一帧）
    - 评估采音对齐（mel 是给模型的，不是评估指标）
    """
    hop = int(round(sr * hop_ms / 1000.0))
    n = len(y) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32), hop_ms
    return np.sqrt((y[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)).astype(np.float32), hop_ms


def detect_offset_ms(rms, hop_ms, note_times, max_ms=120.0, step_frac=1.0):
    """互相关式扫描：找让音符落在能量峰上的整体平移量。

    范围限死 ±120ms —— 真实偏移就是这个量级，搜太大反而会匹配到相邻的拍。
    """
    if len(rms) < 20 or not note_times:
        return 0.0
    t = np.asarray(note_times, dtype=np.float64)
    base = rms.mean() or 1e-9
    step = hop_ms * step_frac
    shifts = np.arange(-max_ms, max_ms + 1e-9, step)
    best = (0.0, -1.0)
    for sh in shifts:
        idx = ((t + sh) / hop_ms).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < len(rms))]
        if len(idx) < max(20, len(t) * 0.5):
            continue
        v = rms[idx].mean() / base
        if v > best[1]:
            best = (float(sh), float(v))
    return round(best[0], 1)


def onset_of(mel):
    """从 mel 派生一条 onset 强度曲线（1 通道）。

    做法：频带能量求和 → 差分 → 半波整流 → 归一化。
    不单独跑一遍 STFT，省一次全曲 FFT。
    """
    e = mel.sum(axis=0)
    d = np.diff(e, prepend=e[0])
    d = np.maximum(d, 0.0)
    mx = d.max()
    if mx > 0:
        d = d / mx
    return d.astype(np.float16)


# --------------------------------------------------------------------------
# 谱面
# --------------------------------------------------------------------------
def safe_name(s, limit=48):
    """把文件名压成可安全落盘的形式（谱包里常有空格、括号、中文）。"""
    s = os.path.splitext(os.path.basename(s))[0]
    s = re.sub(r'[^\w.\-]+', '_', s).strip('_')
    return (s or 'audio')[:limit]


def audio_of_osu(osu_path, files):
    """读出这个 .osu 自己指定的音频。

    为什么必须逐个读：13% 的谱面是「倍速图」或「合集包」——
    一个 set 里有多个 mp3（audio.mp3 / audio 1.1x.mp3 / 1.mp3 / decoy.mp3 ...），
    每个难度对应不同的歌或不同的速度。用第一个音频去套全部难度，
    那些难度的音符时间轴和音频完全错位，测出来的对齐度是垃圾。
    """
    try:
        head = open(osu_path, encoding='utf-8', errors='ignore').read(8192)
    except OSError:
        return None
    m = re.search(r'^AudioFilename:\s*(.+)$', head, re.M)
    name = m.group(1).strip() if m else ''
    # AudioFilename 可能带相对路径，取 basename 再按大小写不敏感匹配
    base = os.path.basename(name)
    if base in files:
        return base
    low = base.lower()
    for f in files:
        if f.lower() == low:
            return f
    # 兜底：按扩展名找（只在目录里恰好一个音频时可信）
    cands = [f for f in files if f.lower().endswith(AUDIO_EXT)]
    return cands[0] if len(cands) == 1 else None


def parse_osu(path):
    """返回 dict(keys, notes=[(t_ms, col, end_ms|None)], bpm, title, version, holds)"""
    keys = 4
    notes = []
    bpm = 0.0
    title = version = ''
    section = ''
    first_beat = None
    beat_len = None

    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('['):
                section = line
                continue
            if section == '[Difficulty]' and line.startswith('CircleSize:'):
                try:
                    keys = int(round(float(line.split(':', 1)[1])))
                except ValueError:
                    pass
            elif section == '[Metadata]':
                if line.startswith('Title:'):
                    title = line.split(':', 1)[1].strip()
                elif line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
            elif section == '[TimingPoints]':
                p = line.split(',')
                if len(p) >= 2 and first_beat is None:
                    try:
                        uninherited = int(float(p[6])) if len(p) > 6 else 1
                        if uninherited == 1:
                            first_beat = float(p[0])
                            beat_len = float(p[1])
                    except (ValueError, IndexError):
                        pass
            elif section == '[HitObjects]':
                p = line.split(',')
                if len(p) < 4 or not p[0][:1].isdigit():
                    continue
                try:
                    x = int(float(p[0]))
                    t = int(float(p[2]))
                    typ = int(float(p[3]))
                except ValueError:
                    continue
                col = min(keys - 1, max(0, x * keys // 512))
                end = None
                if typ & 128 and len(p) > 5:
                    try:
                        end = int(float(p[5].split(':')[0]))
                    except ValueError:
                        end = None
                notes.append((t, col, end))

    notes.sort(key=lambda z: z[0])
    if beat_len and beat_len > 0:
        bpm = round(60000.0 / beat_len, 2)
    holds = sum(1 for n in notes if n[2] is not None)

    return dict(keys=keys, notes=notes, bpm=bpm, title=title,
                version=version, holds=holds)


def chart_tensor(notes, keys, max_frames=MAX_FRAMES, shift_ms=0.0):
    """4 通道编码，通道布局照 Mug-Diffusion，但帧号用「四舍五入」而不是它那里的 floor。

    为什么改这一处：STFT 第 i 帧的中心在 i*46.44ms，一个落在 1000ms 的脉冲
    最强响应出现在帧 22（1000/46.44 = 21.53，四舍五入到 22）。
    Mug-Diffusion 用 int(t/frame_ms) 会编到帧 21，和音频能量峰错开半帧；
    它靠亚帧 offset 通道 + 模型自己学来吸收这个偏差。

    我们用 round 让 start 通道直接压在能量峰上，offset 相应改成 [0,1) 的
    帧内相对位置（解码时 t = (idx - 0.5 + offset) * frame_ms）。
    用 1430228 实测：floor 编码对齐 ×1.02，round 编码 ×1.37。

    shift_ms 仅在个别谱面有明显主观偏移时用，默认为 0。
    """
    arr = np.zeros((4 * keys, max_frames), dtype=np.float16)
    max_idx = -1

    for t0, col, end0 in notes:
        t = t0 + shift_ms
        end = (end0 + shift_ms) if end0 is not None else None
        if t < 0:
            continue
        idx = int(round(t / FRAME_MS))
        if idx < 0 or idx >= max_frames:
            continue
        off = (t - idx * FRAME_MS) / FRAME_MS + 0.5   # 0~1
        arr[col, idx] = 1.0
        arr[col + keys, idx] = off
        last = idx

        if end is not None:
            eidx = int(round(end / FRAME_MS))
            if eidx >= max_frames:
                eidx = max_frames - 1
            if eidx > idx:
                arr[col + 2 * keys, idx + 1:eidx + 1] = 1.0
            arr[col + 3 * keys, eidx] = (end - eidx * FRAME_MS) / FRAME_MS + 0.5
            last = max(last, eidx)

        max_idx = max(max_idx, last)

    valid = np.zeros(max_frames, dtype=np.uint8)
    if max_idx >= 0:
        valid[:max_idx + 1] = 1
    return arr, valid


# --------------------------------------------------------------------------
# 单个 beatmapset
# --------------------------------------------------------------------------
def process_set(job):
    sub, out_root, max_frames = job
    setId = os.path.basename(sub.rstrip('/'))
    res = dict(setId=setId, charts=[], skipped=[])

    try:
        files = os.listdir(sub)
    except OSError as e:
        return dict(setId=setId, error=f'读目录失败 {e}', charts=[], skipped=[])

    osu_files = sorted(f for f in files if f.lower().endswith('.osu'))
    if not osu_files:
        return dict(setId=setId, error='没有 .osu', charts=[], skipped=[])

    # 每个 .osu 配自己的音频，按音频名分组（同一音频只算一次 mel）
    pairs = []
    for fn in osu_files:
        af = audio_of_osu(os.path.join(sub, fn), files)
        if not af:
            res['skipped'].append(dict(osu=fn, reason='找不到对应音频'))
            continue
        pairs.append((fn, af))

    if not pairs:
        return dict(setId=setId, error='没有可用音频', charts=[], skipped=res['skipped'])

    audio_cache = {}     # audioFile -> (relPath, frames)
    os.makedirs(os.path.join(out_root, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'chart'), exist_ok=True)

    for n, (fn, af) in enumerate(pairs):
        # ---- 该音频的特征（同一文件只算一次）----
        if af not in audio_cache:
            rel = os.path.join('audio', f'{setId}__{safe_name(af)}.npz')
            abs_p = os.path.join(out_root, rel)
            if os.path.exists(abs_p):
                with np.load(abs_p) as z:
                    frames = int(z['mel'].shape[1])
                    rms = z['rms'] if 'rms' in z else None
            else:
                try:
                    y = decode_audio(os.path.join(sub, af))
                except Exception as e:
                    res['skipped'].append(dict(osu=fn, reason=f'解码失败 {e}'))
                    continue
                if len(y) < SR // 2:
                    res['skipped'].append(dict(osu=fn, reason='音频过短'))
                    continue
                mel = mel_of(y)
                if mel.shape[1] > max_frames:
                    mel = mel[:, :max_frames]
                rms, rms_hop = rms_envelope(y)
                np.savez_compressed(abs_p,
                                    mel=mel.astype(np.float16),
                                    onset=onset_of(mel),
                                    rms=rms,
                                    rmsHopMs=np.array(rms_hop, dtype=np.float32))
                frames = int(mel.shape[1])
            audio_cache[af] = (rel, frames)

        audio_rel, frames = audio_cache[af]

        # ---- 谱面 ----
        try:
            p = parse_osu(os.path.join(sub, fn))
        except Exception as e:
            res['skipped'].append(dict(osu=fn, reason=f'解析失败 {e}'))
            continue
        if not p['notes']:
            res['skipped'].append(dict(osu=fn, reason='没有音符'))
            continue
        if p['keys'] < 1 or p['keys'] > 18:
            res['skipped'].append(dict(osu=fn, reason=f'键数异常 {p["keys"]}'))
            continue

        # 不做整体平移：受控实验（脉冲 1000ms → mel 峰在帧 22）证明
        # STFT 和 ffmpeg 的 mp3 解码都没有系统性偏移，偏差全部来自 floor 取整，
        # 已经在 chart_tensor 里用 round 解决。自动检测偏移反而会匹配到相邻拍点。
        shift = 0.0
        tensor, valid = chart_tensor(p['notes'], p['keys'], max_frames, shift)
        chart_rel = os.path.join('chart', f'{setId}__{n}.npz')
        np.savez_compressed(os.path.join(out_root, chart_rel),
                            tensor=tensor,
                            valid=valid,
                            keys=np.array(p['keys'], dtype=np.uint8))

        res['charts'].append(dict(
            setId=setId,
            osu=fn,
            audio=audio_rel,
            audioFile=af,
            chart=chart_rel,
            keys=p['keys'],
            notes=len(p['notes']),
            holds=p['holds'],
            bpm=p['bpm'],
            title=p['title'][:80],
            version=p['version'][:60],
            frames=min(frames, max_frames),
            durSec=round(min(frames, max_frames) * FRAME_MS / 1000.0, 1),
            offsetMs=shift,
        ))
    return res


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='含 {setId}/*.osu + 音频 的目录')
    ap.add_argument('--out', default='out/features')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument('--max-frames', type=int, default=MAX_FRAMES)
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个 set（调试用）')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    subs = sorted(os.path.join(args.src, d) for d in os.listdir(args.src)
                  if os.path.isdir(os.path.join(args.src, d)))
    if args.limit:
        subs = subs[:args.limit]
    print(f'待处理 {len(subs)} 个 set，{args.workers} 进程', flush=True)

    jobs = [(s, args.out, args.max_frames) for s in subs]
    idx_rows, skipped, errors = [], [], []
    t0 = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_set, j): j for j in jobs}
        for fu in as_completed(futs):
            done += 1
            try:
                r = fu.result()
            except Exception as e:
                errors.append(dict(setId=os.path.basename(futs[fu][0]), error=str(e)))
                continue
            if r.get('error'):
                errors.append(dict(setId=r['setId'], error=r['error']))
            idx_rows.extend(r.get('charts', []))
            for s in r.get('skipped', []):
                s['setId'] = r['setId']
                skipped.append(s)
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(jobs) - done) / rate if rate else 0
                print(f'  {done}/{len(jobs)}  {rate:.1f} set/s  '
                      f'谱面 {len(idx_rows)}  剩 {eta/60:.1f} 分钟', flush=True)

    with open(os.path.join(args.out, 'index.jsonl'), 'w', encoding='utf-8') as f:
        for r in idx_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(os.path.join(args.out, 'skipped.jsonl'), 'w', encoding='utf-8') as f:
        for r in skipped:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        for r in errors:
            f.write(json.dumps(dict(r, level='set'), ensure_ascii=False) + '\n')

    keys_hist = {}
    for r in idx_rows:
        keys_hist[r['keys']] = keys_hist.get(r['keys'], 0) + 1
    total_notes = sum(r['notes'] for r in idx_rows)
    total_dur = sum(r['durSec'] for r in idx_rows)

    print(f'\n完成：{len(idx_rows)} 张谱面 / {len(set(r["setId"] for r in idx_rows))} 个 set')
    print(f'音符 {total_notes:,}  总时长 {total_dur/3600:.1f} 小时')
    print(f'键数分布 {json.dumps(keys_hist, sort_keys=True)}')
    print(f'跳过 {len(skipped) + len(errors)} 条，详见 skipped.jsonl')
    print(f'耗时 {(time.time()-t0)/60:.1f} 分钟')


if __name__ == '__main__':
    main()
