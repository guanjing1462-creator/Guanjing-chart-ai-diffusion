#!/usr/bin/env python3
"""构建大规模音频条件（踩音）窗口数据集：4K mania 谱面 × 自带音频。

网格：谱面 [TimingPoints] 中第一条 uninherited=1 的 beatLength 定 1/32 格
      step_ms = beatLength / 8，起点 = 该红线 offset。
特征：STFT n_fft=512/hop=128 → 128 带 mel(0-8k) → log1p → 正差分
      → 0-200/200-2000/2000-8000/总 四带求和 → 每格取 max → 每首每通道 P99 归一。
窗口：[4,256]，stride 128；保留 音符数≥16 且 条件覆盖率≥90% 的窗口。

输出 out/diff-data-audio-big/dataset.npz
    X uint8[N,4,256]  每键每格是否有音符（含 LN 持续段）
    C float16[N,256,4] 四带 onset 强度（0-1）
    song int16 / chart_id U32 / split int8 / start_cell int32 / bpm float32
    songs (JSON: id→歌名) / manifest (JSON: 每张谱的记录)
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 22050
N_FFT, HOP = 512, 128
WIN, STRIDE = 256, 128
MIN_NOTES = 16
MIN_COV = 0.90
MAX_REDLINES = 3

# 复用参考实现的 mel/STFT 参数与归一化（不改那个文件，只 import）
_spec = importlib.util.spec_from_file_location(
    'acd', os.path.join(ROOT, 'tools', 'audio-cond-data.py'))
Acd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Acd)


# ---------------- .osu 解析 ----------------
def parse_osu(path):
    """返回 dict(mode, cs, audio, redlines[(time,beatLength)], notes[(abs_ms,col,end_ms)])"""
    mode = cs = None
    audio = ''
    title = version = ''
    redlines = []
    notes = []
    sec = ''
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            if line.startswith('[') and line.endswith(']'):
                sec = line
                continue
            if sec == '[General]':
                if line.startswith('AudioFilename:'):
                    audio = line.split(':', 1)[1].strip()
                elif line.startswith('Mode:'):
                    try:
                        mode = int(float(line.split(':', 1)[1].strip()))
                    except Exception:
                        pass
            elif sec == '[Difficulty]':
                if line.startswith('CircleSize:'):
                    try:
                        cs = float(line.split(':', 1)[1].strip())
                    except Exception:
                        pass
            elif sec == '[Metadata]':
                if line.startswith('Title:'):
                    title = line.split(':', 1)[1].strip()
                elif line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
            elif sec == '[TimingPoints]':
                p = line.split(',')
                if len(p) < 8:
                    continue
                try:
                    t = float(p[0])
                    bl = float(p[1])
                    un = int(float(p[6]))
                except Exception:
                    continue
                if un == 1 and bl > 0:
                    redlines.append((t, bl))
            elif sec == '[HitObjects]':
                p = line.split(',')
                if len(p) < 5:
                    continue
                try:
                    x = int(float(p[0]))
                    t = float(p[2])
                    typ = int(p[3])
                except Exception:
                    continue
                col = min(3, max(0, int(x * 4 / 512)))
                if typ & 128:                                  # LN：取终点
                    try:
                        end = float(p[5].split(':')[0])
                    except Exception:
                        end = t
                    notes.append((t, col, max(t, end)))
                else:
                    notes.append((t, col, t))
    return dict(mode=mode, cs=cs, audio=audio, redlines=redlines, notes=notes,
                title=title, version=version)


def note_times(path):
    """验证脚本用：返回 (音符绝对时间 ms 数组, 首红线 offset ms, beatLength)"""
    d = parse_osu(path)
    if not d['redlines']:
        return np.zeros(0), 0.0, 0.0
    red = sorted(d['redlines'])
    return np.array([n[0] for n in d['notes']], dtype=float), red[0][0], red[0][1]


# ---------------- 音频特征 ----------------
def onset_bands(y, sr=SR):
    """同 tools/audio-cond-data.py：4 通道 onset 包络，返回 [4, F] float32。"""
    return Acd.onset_bands(y, sr)


def grid_features(env, offset_ms, step_ms, n_cells, dur_ms, sr=SR):
    """把帧级包络聚到网格（格内 max），并给出每格有效性（时刻落在音频内）。"""
    feat = np.zeros((n_cells, 4), dtype=np.float32)
    valid = np.zeros(n_cells, dtype=bool)
    if env.shape[1] > 0 and n_cells > 0:
        t_frame = ((np.arange(env.shape[1]) + 1) * HOP + N_FFT // 2) / sr * 1000.0
        t0 = offset_ms + np.arange(n_cells, dtype=np.float64) * step_ms
        t1 = t0 + step_ms
        i0 = np.searchsorted(t_frame, t0, side='left')
        i1 = np.searchsorted(t_frame, t1, side='left')
        for k in range(n_cells):
            a, b = i0[k], i1[k]
            if a >= env.shape[1]:
                continue
            if b <= a:
                b = a + 1
            feat[k] = env[:, a:min(b, env.shape[1])].max(axis=1)
            valid[k] = (t0[k] >= 0.0) and (t0[k] < dur_ms)
    return feat, valid


def norm_p99(x):
    sc = np.zeros(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        p = float(np.percentile(x[:, j], 99)) if x.shape[0] else 0.0
        sc[j] = p if p > 1e-9 else 1.0
    return np.clip(x / sc, 0.0, 1.0), sc


def build_windows(notes, offset_ms, step_ms, n_cells):
    """返回 X[N,4,WIN] uint8, start_cell[N], note_counts[N]。"""
    xs, starts, counts = [], [], []
    if n_cells < WIN:
        return np.zeros((0, 4, WIN), np.uint8), np.zeros(0, np.int32), np.zeros(0, np.int32)
    occ = np.zeros((4, n_cells), dtype=np.uint8)
    for t, col, end in notes:
        c0 = int(np.floor((t - offset_ms) / step_ms))
        c1 = int(np.floor((end - offset_ms) / step_ms))
        c0 = max(c0, 0)
        c1 = max(c1, c0)
        occ[col, min(c0, n_cells - 1)] = 1
        for c in range(c0 + 1, min(c1, n_cells - 1) + 1):
            occ[col, c] = 1
    head = np.zeros(n_cells, dtype=np.int32)
    for t, col, end in notes:
        c0 = int(np.floor((t - offset_ms) / step_ms))
        if 0 <= c0 < n_cells:
            head[c0] += 1
    csum = np.concatenate([[0], np.cumsum(head)])
    for s in range(0, n_cells - WIN + 1, STRIDE):
        xs.append(occ[:, s:s + WIN])
        starts.append(s)
        counts.append(csum[s + WIN] - csum[s])
    if not xs:
        return np.zeros((0, 4, WIN), np.uint8), np.zeros(0, np.int32), np.zeros(0, np.int32)
    return np.stack(xs), np.asarray(starts, np.int32), np.asarray(counts, np.int32)


def assign_splits(counts_by_set, ratios=(0.80, 0.10, 0.10)):
    """按窗口数把「整个 set」贪心分到 train/val/test，保证同 set 不跨集。

    counts_by_set: {set_id: n_windows}
    返回 {set_id: 0|1|2}
    """
    total = float(sum(counts_by_set.values())) or 1.0
    target = [total * r for r in ratios]
    cur = [0.0, 0.0, 0.0]
    out = {}
    for sid, w in sorted(counts_by_set.items(), key=lambda kv: (-kv[1], kv[0])):
        j = max(range(3), key=lambda k: target[k] - cur[k])
        out[sid] = j
        cur[j] += w
    return out


def resplit(out_dir):
    """读已有 dataset.npz + manifest.json，按窗口数重算均衡 split 后写回。"""
    man_path = os.path.join(out_dir, 'manifest.json')
    meta_path = os.path.join(out_dir, 'meta.json')
    npz_path = os.path.join(out_dir, 'dataset.npz')
    man = json.load(open(man_path, encoding='utf-8'))
    counts = {}
    for m in man:
        counts[m['SetID']] = counts.get(m['SetID'], 0) + m['windows']
    split_of = assign_splits(counts)
    d = dict(np.load(npz_path))
    old = d['split']
    new = np.zeros_like(old)
    pos = 0
    for m in man:
        new[pos:pos + m['windows']] = split_of[m['SetID']]
        m['split'] = int(split_of[m['SetID']])
        pos += m['windows']
    assert pos == len(old), f'窗口数与 manifest 不符 {pos} vs {len(old)}'
    d['split'] = new
    np.savez_compressed(npz_path, **d)
    json.dump(man, open(man_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding='utf-8'))
        meta['manifest'] = man
        meta['split_counts'] = {str(k): int((new == k).sum()) for k in (0, 1, 2)}
        json.dump(meta, open(meta_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('重划分后 split：', {k: int((new == k).sum()) for k in (0, 1, 2)},
          ' 共', len(new), '窗口')
    print('涉及 set 数：', {k: sum(1 for v in split_of.values() if v == k) for k in (0, 1, 2)})
    print('变更窗口：', int((old != new).sum()))
    return 0


def fixids(out_dir, dl_dir):
    """chart_id 是 U32（32 字符），长难度名截断会撞名；改成 SetID-两位序号-难度名。"""
    npz_path = os.path.join(out_dir, 'dataset.npz')
    man_path = os.path.join(out_dir, 'manifest.json')
    meta_path = os.path.join(out_dir, 'meta.json')
    man = json.load(open(man_path, encoding='utf-8'))
    d = dict(np.load(npz_path))
    old = d['chart_id']
    new = np.empty(len(old), dtype='U32')
    pos, per_set = 0, {}
    for m in man:
        sid = m['SetID']
        k = per_set.get(sid, 0)
        per_set[sid] = k + 1
        ver = ''
        p = os.path.join(dl_dir, str(sid), m['chart'])
        if os.path.exists(p):
            for line in open(p, encoding='utf-8-sig', errors='replace'):
                if line.startswith('Version:'):
                    ver = line.split(':', 1)[1].strip()
                    break
        cid = '{}-{:02d}-{}'.format(sid, k, (ver or 'diff')[:14])[:32]
        new[pos:pos + m['windows']] = cid
        m['chart_id'] = cid
        pos += m['windows']
    assert pos == len(old)
    d['chart_id'] = new
    np.savez_compressed(npz_path, **d)
    json.dump(man, open(man_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding='utf-8'))
        meta['manifest'] = man
        json.dump(meta, open(meta_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    u_old, u_new = len(set(old.tolist())), len(set(new.tolist()))
    print(f'chart_id 唯一数 {u_old} → {u_new}（谱面 {len(man)} 张）')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dl', default=os.path.join(ROOT, 'out', 'audio-dl'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'diff-data-audio-big'))
    ap.add_argument('--limit-sets', type=int, default=0)
    ap.add_argument('--fixids', action='store_true',
                    help='重写 chart_id：SetID-序号-难度名（避免 U32 截断重名）')
    ap.add_argument('--resplit-only', action='store_true',
                    help='只按窗口数重算均衡 split（不改 X/C）')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.resplit_only:
        return resplit(a.out)
    if a.fixids:
        return fixids(a.out, a.dl)

    sets = sorted(int(d) for d in os.listdir(a.dl)
                  if d.isdigit() and os.path.isdir(os.path.join(a.dl, d)))
    if a.limit_sets:
        sets = sets[:a.limit_sets]
    print(f'待处理谱面包 {len(sets)} 个', flush=True)

    # set 级 split：先收集，构建完再按「窗口数」均衡划分（同 set 绝不跨集）
    split_of = {}

    fname_map = {}
    if os.path.exists(os.path.join(a.dl, '_fetch_log.json')):
        lg = json.load(open(os.path.join(a.dl, '_fetch_log.json')))
        for k, v in lg.get('ok', {}).items():
            fname_map[int(k)] = (v.get('title', ''), v.get('artist', ''))

    Xs, Cs, songs, cids, sps, sts, bpms = [], [], [], [], [], [], []
    set_of_chunk = []
    manifest = []
    skip = {'多红线': [], '无红线': [], '无音频': [], '读谱失败': [], '非4K': [], '时长不足': []}
    songs_meta = {}
    song_idx = 0

    t_all = time.time()
    for si, sid in enumerate(sets):
        d = os.path.join(a.dl, str(sid))
        files = os.listdir(d)
        osus = sorted(f for f in files if f.lower().endswith('.osu'))
        auds = [f for f in files if f.lower().endswith(('.mp3', '.ogg', '.wav'))]
        if not auds:
            skip['无音频'].append(sid)
            continue

        charts = []
        for f in osus:
            try:
                info = parse_osu(os.path.join(d, f))
            except Exception:
                skip['读谱失败'].append(f'{sid}/{f}')
                continue
            if info['mode'] != 3 or info['cs'] is None or int(round(info['cs'])) != 4:
                skip['非4K'].append(f'{sid}/{f}')
                continue
            charts.append((f, info))
        if not charts:
            continue

        # 音频：每张谱用自己的 AudioFilename（多曲 pack 里每首谱音频不同！）
        import librosa
        default_audio = os.path.join(
            d, max(auds, key=lambda z: os.path.getsize(os.path.join(d, z))))
        env_cache = {}

        def get_env(path):
            """返回 (env, dur_ms)；缓存最多 3 条音频的包络。"""
            if path in env_cache:
                return env_cache[path]
            try:
                y, _ = librosa.load(path, sr=SR, mono=True)
            except Exception:
                env_cache[path] = None
                return None
            y = np.asarray(y, dtype=np.float32)
            if len(y) < SR * 30:
                env_cache[path] = None
                return None
            dm = len(y) / SR * 1000.0
            e = onset_bands(y)
            del y
            if len(env_cache) >= 3:
                env_cache.clear()
            env_cache[path] = (e, dm)
            return env_cache[path]

        title, artist = fname_map.get(sid, (f'Set{sid}', ''))
        n_win_set = 0
        for f, info in charts:
            red = sorted(info['redlines'])
            if not red:
                skip['无红线'].append(f'{sid}/{f}')
                continue
            if len(red) > MAX_REDLINES:
                skip['多红线'].append(f'{sid}/{f}')
                continue
            cand = info['audio']
            bn = os.path.basename(cand) if cand else ''
            ap_ = os.path.join(d, bn) if bn in auds else default_audio
            got = get_env(ap_)
            if got is None:
                skip['时长不足'].append(f'{sid}/{f}')
                continue
            env, dur_ms = got
            offset_ms, beat_len = red[0]
            step_ms = beat_len / 8.0                        # 1/32 拍
            n_cells = int(max(0, np.floor((dur_ms - offset_ms) / step_ms)))
            if n_cells < WIN:
                continue
            feat_raw, valid = grid_features(env, offset_ms, step_ms, n_cells, dur_ms)
            feat, scale = norm_p99(feat_raw)
            X, starts, counts = build_windows(info['notes'], offset_ms, step_ms, n_cells)
            if X.shape[0] == 0:
                continue
            # 窗口级过滤：音符数≥16 且条件覆盖率≥90%
            keep = []
            for i, s in enumerate(starts):
                cov = float(valid[s:s + WIN].mean())
                if counts[i] >= MIN_NOTES and cov >= MIN_COV:
                    keep.append(i)
            if not keep:
                continue
            keep = np.asarray(keep)
            _v = info['version'] or os.path.splitext(f)[0]
            cid = '{}-{}'.format(sid, _v[:20])
            Xs.append(X[keep])
            Cs.append(feat[starts[keep][:, None] + np.arange(WIN)[None, :]].astype(np.float16))
            nw = len(keep)
            songs.append(np.full(nw, song_idx, np.int16))
            cids.append(np.array([cid] * nw))
            sps.append(np.full(nw, 0, np.int8))
            set_of_chunk.append(sid)
            sts.append(starts[keep].astype(np.int32))
            bpms.append(np.full(nw, 60000.0 / beat_len, np.float32))
            n_win_set += nw
            manifest.append({
                'SetID': sid, 'chart': f, 'chart_id': cid,
                'title': title, 'artist': artist,
                'redlines': len(red), 'beat_len': round(beat_len, 4),
                'step_ms': round(step_ms, 4), 'offset_ms': round(offset_ms, 3),
                'bpm': round(60000.0 / beat_len, 3),
                'notes': len(info['notes']), 'windows': nw,
                'split': 0,
                'audio': os.path.basename(ap_),
            })
        if n_win_set:
            songs_meta[str(song_idx)] = f'{title} [{sid}]' if title else f'Set{sid}'
            song_idx += 1
        if (si + 1) % 5 == 0:
            print(f'  [{si+1}/{len(sets)}] 谱面 {len(manifest)}，窗口 {sum(len(x) for x in Xs)}，'
                  f'{time.time()-t_all:.0f}s', flush=True)

    if not Xs:
        print('没有产出任何窗口', file=sys.stderr)
        return 1
    X = np.concatenate(Xs).astype(np.uint8)
    C = np.concatenate(Cs).astype(np.float16)
    song = np.concatenate(songs).astype(np.int16)
    cid = np.concatenate(cids).astype('U32')
    split = np.concatenate(sps).astype(np.int8)
    start_cell = np.concatenate(sts).astype(np.int32)
    bpm = np.concatenate(bpms).astype(np.float32)

    # 按窗口数均衡划分 split（同 set 整体进同一个 split）
    counts_by_set = {}
    for sid, arr in zip(set_of_chunk, Xs):
        counts_by_set[sid] = counts_by_set.get(sid, 0) + len(arr)
    split_of = assign_splits(counts_by_set)
    sps = [np.full(len(x), split_of[sid], np.int8) for sid, x in zip(set_of_chunk, Xs)]
    for m in manifest:
        m['split'] = int(split_of[m['SetID']])
    split = np.concatenate(sps).astype(np.int8)      # 覆盖默认的 0

    out_npz = os.path.join(a.out, 'dataset.npz')
    np.savez_compressed(out_npz, X=X, C=C, song=song, chart_id=cid,
                        split=split, start_cell=start_cell, bpm=bpm)
    meta = {
        'note': '4K mania 谱面 × osu.direct 自带音频，1/32 网格对齐的音频条件窗口',
        'grid': {'rule': '首条 uninherited=1 的 beatLength / 8 毫秒',
                 'origin': '该红线的 offset',
                 'max_redlines': MAX_REDLINES},
        'features': {'channels': Acd.CH_NAMES, 'sr': SR, 'n_fft': N_FFT, 'hop': HOP,
                     'n_mels': Acd.N_MELS, 'fmax': Acd.FMAX,
                     'pipeline': 'STFT→mel→log1p→正差分→四带求和→格内max→每首每通道P99归一'},
        'window': {'shape': [4, WIN], 'stride': STRIDE,
                   'min_notes': MIN_NOTES, 'min_coverage': MIN_COV},
        'n_windows': int(X.shape[0]), 'n_charts': len(manifest),
        'n_songs': int(song_idx), 'n_sets': len({m['SetID'] for m in manifest}),
        'split_counts': {str(k): int((split == k).sum()) for k in (0, 1, 2)},
        'songs': songs_meta, 'manifest': manifest, 'skipped': skip,
    }
    with open(os.path.join(a.out, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)

    print(f'\n数据集：{out_npz}  {os.path.getsize(out_npz)/1e6:.1f} MB')
    print(f'X {X.shape} uint8 / C {C.shape} float16')
    print(f'谱面 {len(manifest)} 张 / 歌 {song_idx} 首 / 窗口 {X.shape[0]}')
    print(f'split 分布: ' + str(meta['split_counts']))
    print(f'跳过: ' + ', '.join(f'{k}={len(v)}' for k, v in skip.items() if v))
    print(f'用时 {time.time()-t_all:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
