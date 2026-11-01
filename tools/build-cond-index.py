#!/usr/bin/env python3
"""把 out/diff-data/dataset.npz 的窗口对到「有音频的歌 + 起始格」，产出音频条件索引。

为什么需要「溯源」：dataset.npz 只存了 X / style / diff，没有记录每个窗口来自哪张谱。
好在它的窗口是 tools/diff-data.py 那套切片规则生成的，可以**重放**同一套规则
（同一批 out/packs/*/*.osu、同样的中位步长/滑窗/密度阈值），按窗口内容的 md5
把 dataset 的每一行找回来。重放完全对不上时脚本会报错而不是瞎猜。

匹配规则（三条都过才算这首歌的窗口）：
  1) 谱面元数据（Title/TitleUnicode/Artist/ArtistUnicode/AudioFilename/Version）
     命中该歌的名字（音频文件名归一化后的名字或 " - " 之后的名字）；
  2) 谱面 1/16 步长（音符相邻间隔中位数）与该歌分析文件的 stepMs 相差 ≤ 3%；
  3) 谱面音符落在该歌网格线上的比例 ≥ 0.6（相位 ±15% 容差）。
  另外把「音符处的 onset 特征 vs 随机基线」记为音频佐证（-- 只作诊断：
  谱面密时该比值天然接近 1，不足以单独判定，所以不作门槛）。
  然后 start = round((窗口首格时刻 base_t - offsetMs) / stepMs)，
  要求 0 ≤ start ≤ steps - 128（窗口正好 128 格）。

输出：
  out/diff-data/cond-index.npz   键 idx / song / start（int32 / int16 / int32）
  out/diff-data/cond-index.json  统计、每首歌的匹配谱面与诊断、song id → 歌名
  （--labeled，默认开）out/diff-data/cond-windows.npz
      直接从这 6 首的 GJ 谱在分析网格上切窗：X[W,4,128] uint8 + song + start，
      这样即使 dataset.npz 里一个都匹配不上，也能先把音频条件分支的接口跑通。

⚠ 不修改 dataset.npz；小样本，仅用于接口验证。
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

WIN = 128                 # 窗口 128 个 1/16
STYLES = ['tech', 'jack', 'speed', 'stamina']
STEP_TOL = 0.03           # 谱面步长与分析 stepMs 的允许相对误差
PHASE_MIN = 0.60          # 音符落网格线的最低比例
PHASE_TOL = 0.15          # "落线"判定：相位在 ±15% 格内
AUDIO_LINK_MIN = 1.03     # 音符所在格的 onset 特征 / 随机基线 的最低比值（低于此说明谱不是这条音频的）


# ---------------------------------------------------------------- .osu 解析
def parse_notes(path):
    """返回 [(time_ms, [cols])]。cols: 0..3 对应 x=64/192/320/448。"""
    xs = {64: 0, 192: 1, 320: 2, 448: 3}
    by_t = {}
    for ln in open(path, encoding='utf-8', errors='ignore'):
        p = ln.split(',')
        if len(p) < 5:
            continue
        try:
            x, t, typ = int(p[0]), int(p[2]), int(p[3])
        except ValueError:
            continue
        if x not in xs:
            continue
        if typ & 1 or typ & 128:
            by_t.setdefault(t, set()).add(xs[x])
    return sorted((t, sorted(c)) for t, c in by_t.items())


def parse_header(path):
    """只读到 [HitObjects] 之前的元数据（快）。"""
    fields = {}
    for ln in open(path, encoding='utf-8', errors='ignore'):
        s = ln.strip()
        if s.startswith('[HitObjects]'):
            break
        if ':' not in s:
            continue
        k, v = s.split(':', 1)
        k = k.strip()
        if k in ('Title', 'TitleUnicode', 'Artist', 'ArtistUnicode',
                 'Version', 'AudioFilename', 'Creator', 'BeatmapID', 'Source'):
            fields[k] = v.strip()
    return fields


def median_step(rows):
    """1/16 步长估计：相邻音符间隔的中位数（与 diff-data.py 一致）。"""
    if len(rows) < 3:
        return None
    gaps = np.diff([t for t, _ in rows]).astype(float)
    gaps = gaps[(gaps > 40) & (gaps < 400)]
    if len(gaps) < 20:
        return None
    return float(np.median(gaps))


def roll_matrix(rows, step_ms, offset_ms, start_cell, n=WIN):
    """把音符映射到 [4, n] 网格（格 k 的时间 = offset + (start_cell+k)*step）。"""
    m = np.zeros((4, n), dtype=np.uint8)
    for t, cols in rows:
        k = int(round((t - offset_ms) / step_ms)) - start_cell
        if 0 <= k < n:
            for c in cols:
                m[c, k] = 1
    return m


def win_hash(m):
    return hashlib.md5(m.tobytes()).hexdigest()


# ---------------------------------------------------------------- 重放溯源
def replay_packs(pack_dir='out/packs', songs=(), verbose=True):
    """按 diff-data.py 的规则重放所有窗口。

    返回 (windows, chart_info)：
        windows     [(chart_path, base_t, md5)]
        chart_info  {chart_path: {step, n_notes, phase: {song_id: 比例}}}
    """
    out = []
    chart_info = {}
    files = []
    for style in STYLES:
        files.extend(sorted(glob.glob(os.path.join(pack_dir, style, '*.osu'))))
    if not files:
        raise FileNotFoundError(f'{pack_dir} 下没有谱面，无法重放')
    for i, f in enumerate(files):
        if verbose and i and i % 400 == 0:
            print(f'    重放 {i}/{len(files)} 张，累计窗口 {len(out)}', flush=True)
        rows = parse_notes(f)
        if len(rows) < WIN:
            continue
        step = median_step(rows)
        if step is None:
            continue
        chart_info[f] = {          # 不保留 rows：2608 张谱全留会吃掉几百 MB
            'step': step,
            'n_notes': len(rows),
            'phase': {s['song_id']: phase_alignment(rows, s['offset_ms'], s['step_ms'])
                      for s in songs},
        }
        for start in range(0, len(rows) - 32, WIN // 2):
            seg = rows[start:start + 400]
            base_t = seg[0][0]
            sub = [r for r in seg if r[0] < base_t + step * WIN]
            if len(sub) < 32:
                continue
            m = roll_matrix(sub, step, base_t, 0)
            if m.sum() < 12:
                continue
            out.append((f, base_t, win_hash(m)))
    return out, chart_info


# ---------------------------------------------------------------- 歌与谱面
def song_tokens(audio_relpath):
    """从音频文件名得到用于匹配谱面元数据的名字候选。"""
    stem = os.path.splitext(os.path.basename(audio_relpath))[0]
    cand = {stem}
    if ' - ' in stem:
        cand.add(stem.split(' - ', 1)[1])          # "Artist - Title" → Title
    norm = set()
    for c in cand:
        n = ''.join(ch for ch in c.lower() if ch.isalnum())
        if len(n) >= 4:
            norm.add(n)
    return norm


def _alnum(s):
    return ''.join(ch for ch in s.lower() if ch.isalnum())


def header_matches(tokens, hdr):
    """谱面元数据里是否出现歌名（去空格/标点后子串匹配）。"""
    blob = ' '.join(hdr.get(k, '') for k in
                    ('Title', 'TitleUnicode', 'Artist', 'ArtistUnicode',
                     'AudioFilename', 'Version'))
    flat = _alnum(blob)
    return any(t in flat for t in tokens)


def phase_alignment(rows, offset_ms, step_ms):
    """音符落在网格线上（±15% 格）的比例。"""
    if not rows:
        return 0.0
    t = np.array([r[0] for r in rows], dtype=np.float64)
    ph = ((t - offset_ms) % step_ms) / step_ms
    return float(np.mean((ph < PHASE_TOL) | (ph > 1.0 - PHASE_TOL)))


def audio_link_score(feat_total, rows, offset_ms, step_ms, steps):
    """诊断用：音符所在格的特征值 vs 随机基线（>1 表示特征确实解释了音符）。"""
    if not rows or feat_total is None:
        return 0.0
    k = np.array([int(round((t - offset_ms) / step_ms)) for t, _ in rows])
    k = k[(k >= 0) & (k < steps)]
    if len(k) < 32:
        return 0.0
    rng = np.random.default_rng(0)
    base = feat_total[rng.integers(0, steps, len(k))].mean()
    return float(feat_total[k].mean() / max(base, 1e-9))


def load_songs(analysis_dir='out/multisong'):
    """读 6 首的 fx.json + cond.npz，返回 [{id, name, offset, step, steps, feat}]。"""
    songs = []
    for p in sorted(glob.glob(os.path.join(analysis_dir, '*-fx.json'))):
        sid = os.path.basename(p)[:-len('-fx.json')]
        meta = json.load(open(p, encoding='utf-8'))
        cond = os.path.join(analysis_dir, f'{sid}-cond.npz')
        feat = None
        if os.path.exists(cond):
            feat = np.load(cond)['feat'][:, 3].astype(np.float32)
        songs.append({
            'id': sid,
            'name': sid,
            'audio': meta.get('source', ''),
            'offset_ms': float(meta['offsetMs']),
            'step_ms': float(meta['stepMs']),
            'steps': int(meta['steps']),
            'bpm': meta.get('bpm'),
            'feat_total': feat,
            'cond': cond if feat is not None else None,
        })
    for i, s in enumerate(songs):
        s['song_id'] = i
    return songs


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description='生成音频条件索引（dataset 窗口 → 歌 + 起始格）')
    ap.add_argument('--dataset', default='out/diff-data/dataset.npz')
    ap.add_argument('--pack-dir', default='out/packs')
    ap.add_argument('--analysis-dir', default='out/multisong')
    ap.add_argument('--chart-dir', default='out/multisong/charts',
                    help='这 6 首自己的谱面目录（用于 --labeled 切窗）')
    ap.add_argument('--out', default='out/diff-data/cond-index.npz')
    ap.add_argument('--labeled', default='out/diff-data/cond-windows.npz',
                    help='直接从 6 首的谱面切窗的配套小样本（空串=不生成）')
    ap.add_argument('--stride', type=int, default=WIN // 2, help='--labeled 切窗步长（格）')
    a = ap.parse_args()

    if not os.path.exists(a.dataset):
        print(f'缺少 {a.dataset}', file=sys.stderr)
        return 1
    dz = np.load(a.dataset)
    X = dz['X']
    print(f'数据集 {a.dataset}: X={X.shape} style={dz["style"].shape} diff={dz["diff"].shape}')

    songs = load_songs(a.analysis_dir)
    if not songs:
        print('没有找到任何 *-fx.json', file=sys.stderr)
        return 1
    print(f'有音频的歌 {len(songs)} 首：' + '、'.join(s['id'] for s in songs))

    # 1) 重放 packs，拿每个候选窗口的来源
    print('\n[1/4] 重放 out/packs 的窗口（还原 dataset 每行的来源）…')
    replay, chart_info = replay_packs(a.pack_dir, songs)
    rep_map, hash_map = {}, {}
    for path, base_t, h in replay:
        rep_map.setdefault(path, []).append(base_t)
        hash_map.setdefault(h, []).append((path, base_t))
    del replay
    print(f'    重放得到 {sum(len(v) for v in rep_map.values())} 个窗口，'
          f'来自 {len(rep_map)} 张谱，{len(hash_map)} 个不同内容')

    # 2) 找每首歌的谱面候选并校验
    print('\n[2/4] 在 packs 里找这 6 首的谱面（步长 → 相位 → 音频对齐，元数据作佐证）…')
    song_charts = {s['song_id']: [] for s in songs}
    diagnostics = []
    grid_hits = 0
    meta_hits = 0
    for path, info in chart_info.items():
        hdr = parse_header(path)
        step = info['step']
        rows = None                  # 只有候选谱才解析音符（省内存/时间）
        for s in songs:
            step_ok = abs(step - s['step_ms']) / s['step_ms'] <= STEP_TOL
            phase = info['phase'][s['song_id']]
            phase_ok = phase >= PHASE_MIN
            meta_ok = header_matches(song_tokens(s['audio']), hdr)
            meta_hits += 1 if meta_ok else 0
            if not (step_ok or phase_ok or meta_ok):
                continue
            grid_hits += 1 if (step_ok and phase_ok) else 0
            if rows is None:
                rows = parse_notes(path)
            link = audio_link_score(s['feat_total'], rows, s['offset_ms'], s['step_ms'], s['steps'])
            rec = {
                'chart': path,
                'song': s['id'],
                'title': hdr.get('Title', ''),
                'artist': hdr.get('Artist', ''),
                'version': hdr.get('Version', ''),
                'chart_step_ms': round(step, 3),
                'song_step_ms': round(s['step_ms'], 3),
                'phase_align': round(phase, 3),
                'audio_link': round(link, 3),
                'metadata_match': bool(meta_ok),
                'accepted': False,
                'reason': '',
            }
            if not meta_ok:
                rec['reason'] = ('元数据（标题/艺人/音频名）对不上这首歌 → 不是这首歌的谱；'
                                 + (f'只是 1/16 步长巧合（{step:.2f}ms ≈ {s["step_ms"]:.2f}ms）' if step_ok else ''))
            elif not step_ok:
                rec['reason'] = (f'网格步长不一致（谱面 {step:.2f}ms vs 分析 {s["step_ms"]:.2f}ms，'
                                 f'偏差 {abs(step - s["step_ms"]) / s["step_ms"] * 100:.1f}%）'
                                 '→ 谱面不是对着这条音频/这个细分做的（多半是变速练习包）')
            elif not phase_ok:
                rec['reason'] = f'音符没落在网格线上（相位对齐 {phase:.2f} < {PHASE_MIN}）'
            elif link < AUDIO_LINK_MIN:
                rec['reason'] = (f'谱面音符处的 onset 特征不比随机高（{link:.2f} < {AUDIO_LINK_MIN}）'
                                 '→ 音符没有踩在这条音频的击打上')
            else:
                rec['accepted'] = True
                rec['reason'] = ('步长 + 相位 + 音频对齐都通过'
                                 + ('，元数据也命中' if meta_ok else '（元数据未命中）'))
                song_charts[s['song_id']].append({'path': path, 'rows': rows, 'step': step})
            diagnostics.append(rec)
            flag = '✔' if rec['accepted'] else '✘'
            print(f'    {s["id"]:38s} ← {os.path.basename(path):14s} '
                  f'{str(rec["version"])[:26]:26s} {flag} {rec["reason"]}')
    print(f'    全库筛查：{len(chart_info)} 张谱；步长+相位同时吻合 {grid_hits} 个(谱,歌)组合，'
          f'元数据命中 {meta_hits} 个，最终接受 '
          f'{sum(len(v) for v in song_charts.values())} 个')

    # 3) 把 dataset 的每一行对到 (song, start)
    print('\n[3/4] 对齐 dataset 窗口…')
    row_hash = [win_hash(X[i]) for i in range(X.shape[0])]

    # (chart, base_t) → (song_id, start)
    chart_owner = {}
    for sid, charts in song_charts.items():
        s = next(s for s in songs if s['song_id'] == sid)
        for c in charts:
            chart_owner[c['path']] = (sid, s)

    idx, sng, sta = [], [], []
    unresolved = 0
    ambiguous = 0
    per_song = {s['song_id']: 0 for s in songs}
    for i, h in enumerate(row_hash):
        cands = hash_map.get(h)
        if not cands:
            unresolved += 1
            continue
        picks = []
        for path, bt in cands:
            if path in chart_owner:
                sid, s = chart_owner[path]
                start = int(round((bt - s['offset_ms']) / s['step_ms']))
                if 0 <= start <= s['steps'] - WIN:
                    picks.append((sid, start))
        if not picks:
            continue
        if len(set(picks)) > 1:
            ambiguous += 1
        sid, start = picks[0]
        idx.append(i)
        sng.append(sid)
        sta.append(start)
        per_song[sid] += 1

    idx = np.asarray(idx, dtype=np.int32)
    sng = np.asarray(sng, dtype=np.int16)
    sta = np.asarray(sta, dtype=np.int32)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, idx=idx, song=sng, start=sta)

    print(f'    能对上 {len(idx)} / {X.shape[0]} 个窗口'
          f'（来源无法还原 {unresolved} 个，多来源歧义 {ambiguous} 个）')
    for s in songs:
        print(f'      {s["id"]:38s} {per_song[s["song_id"]]:6d} 个窗口')
    print(f'    → {a.out}')

    # 4) 配套：直接从这 6 首的谱面在分析网格上切窗（接口验证用）
    labeled_info = None
    if a.labeled:
        print('\n[4/4] 生成配套小样本（这 6 首自己的谱面切窗）…')
        Xs, sg, st = [], [], []
        per_song_win = {}
        for s in songs:
            cp = os.path.join(a.chart_dir, f'{s["id"]}.osu')
            if not os.path.exists(cp):
                print(f'    {s["id"]}: 缺谱面 {cp}，跳过')
                continue
            rows = parse_notes(cp)
            n_notes = len(rows)
            cnt = 0
            for start in range(0, max(1, s['steps'] - WIN + 1), a.stride):
                m = roll_matrix(rows, s['step_ms'], s['offset_ms'], start)
                if m.sum() < 12:
                    continue
                # 窗口内至少有 32 个音符时刻（与 diff-data.py 的密度门槛一致）
                if len([1 for t, _ in rows
                        if s['offset_ms'] + start * s['step_ms'] <= t
                        < s['offset_ms'] + (start + WIN) * s['step_ms']]) < 32:
                    continue
                Xs.append(m)
                sg.append(s['song_id'])
                st.append(start)
                cnt += 1
            offgrid = round(1.0 - phase_alignment(rows, s['offset_ms'], s['step_ms']), 3)
            per_song_win[s['id']] = {'windows': cnt, 'notes': n_notes, 'off_grid_ratio': offgrid}
            print(f'    {s["id"]:38s} {n_notes:5d} 音符 → {cnt:4d} 个窗口'
                  f'（{offgrid * 100:.0f}% 音符在 1/32 半格上，切窗时会吸到最近格）')
        if Xs:
            np.savez_compressed(a.labeled,
                                X=np.stack(Xs).astype(np.uint8),
                                song=np.asarray(sg, dtype=np.int16),
                                start=np.asarray(st, dtype=np.int32),
                                # 附带 id→歌名，免得下游还要去别处查
                                songs=np.asarray([s['id'] for s in songs]))
            labeled_info = {'path': a.labeled, 'windows': len(Xs),
                            'per_song': per_song_win, 'stride': a.stride}
            print(f'    → {a.labeled}（{len(Xs)} 个窗口）')

    # 5) 统计落盘
    report = {
        'dataset': a.dataset,
        'dataset_windows': int(X.shape[0]),
        'matched_windows': int(len(idx)),
        'unresolved_provenance': int(unresolved),
        'ambiguous_multi_source': int(ambiguous),
        'per_song_windows': {s['id']: int(per_song[s['song_id']]) for s in songs},
        'song_ids': {s['id']: int(s['song_id']) for s in songs},
        'songs': [{
            'id': s['id'], 'song_id': int(s['song_id']), 'audio': s['audio'],
            'offset_ms': s['offset_ms'], 'step_ms': s['step_ms'], 'steps': s['steps'],
            'bpm': s['bpm'], 'cond_file': s['cond'],
            'matched_charts': [c['path'] for c in song_charts[s['song_id']]],
        } for s in songs],
        'chart_diagnostics': diagnostics,
        'labeled': labeled_info,
        'note': '小样本（6 首有音频）仅用于音频条件分支的接口验证，不是全量训练数据',
    }
    jp = os.path.splitext(a.out)[0] + '.json'
    with open(jp, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f'\n统计：{jp}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
