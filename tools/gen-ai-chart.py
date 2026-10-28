#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 写谱一键生成：可选键型（RICE 纯米 / LN 两大类）+ 输出修复。

用法：
  python3 tools/gen-ai-chart.py --mel /tmp/yamero_mel.npz --style rice-tech \
      --version "AI v3" -o out-dir/

核心修复（回应 [AI v2] 的「全屏 LN 墙」）：
  模型在潜空间里还没学会人类的 LN 长度分布（p50 0.42 拍 / p90 0.83 拍），
  解码出来的是「按住直到遥遥无期」的超长条。polish 按人类分布钳制：
    1. LN < 60ms  → 转 tap（按不住的条不是条）
    2. LN > 1 拍  → 截断到 1/8 格线（4K 14 档没有 1 拍以上的条；
       截断点取 1/8 网格保节奏感，尾留同列 ≥1 格防连按）
    3. 局部 LN 墙：4s 滑窗内 LN 行占比 >55% → 砍该窗内最长的条
       （长条才是墙的主因，人类 LN 成段但窗内占比 ≤45%）
  RICE 大类则把 LN 全部转 tap——「纯米」是用户可选的键型大类。

键型 = 两大类 × 子键型，映射到 Mug 同款 21 token：
  RICE（无 LN）：rc=1 + pattern bool
  LN（带长条）：ln/hb=1 + ln_ratio + pattern bool
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
torch.set_num_threads(4)
import train_diffusion as td  # noqa: E402

BPM = 185.0
BEAT = 60000.0 / BPM            # 324.32ms
CELL = BEAT / 4                 # 1/16 格 81.08ms
LN_MIN_MS = 60.0                # 更短按不住
LN_MAX_BEATS = 1.0              # 4K 14 档最长 1 拍
LN_WALL_WIN_MS = 4000.0         # 局部墙滑窗
LN_WALL_SHARE = 0.55            # 窗内 LN 行占比上限（人类 ≤45%，留余量）

# 键型表：大类 × 子键型 → Mug token。ett 是 Etterna overall（25≈D13-15 档）。
STYLES = {
    # ---- RICE 纯米（零长条）----
    'rice-stream':    {'cls': 'rice', 'tok': {'rc': 1, 'stream': 1, 'ett': 25}},
    'rice-jumpstream': {'cls': 'rice', 'tok': {'rc': 1, 'jumpstream': 1, 'ett': 25}},
    'rice-tech':      {'cls': 'rice', 'tok': {'rc': 1, 'technical': 1, 'ett': 25}},
    'rice-chordjack': {'cls': 'rice', 'tok': {'rc': 1, 'chordjack': 1, 'ett': 25}},
    'rice-stamina':   {'cls': 'rice', 'tok': {'rc': 1, 'stamina': 1, 'ett': 26}},
    # ---- LN 带长条 ----
    'ln-mix':         {'cls': 'ln', 'tok': {'hb': 1, 'technical': 1, 'ln_ratio': 0.12, 'ett': 25}},
    'ln-stream':      {'cls': 'ln', 'tok': {'ln': 1, 'stream': 1, 'ln_ratio': 0.16, 'ett': 25}},
    'ln-tech':        {'cls': 'ln', 'tok': {'hb': 1, 'technical': 1, 'ln_ratio': 0.10, 'ett': 25}},
}


def parse_osu(path):
    txt = open(path, encoding='utf-8').read()
    head, body = txt.split('[HitObjects]')
    objs = [l.split(',') for l in body.strip().splitlines() if l.strip()]
    return head, objs


def write_osu(path, head, objs):
    open(path, 'w', encoding='utf-8').write(
        head + '[HitObjects]\n' + '\n'.join(','.join(o) for o in objs) + '\n')


def col(o):
    return int(o[0]) // 128


def ln_end(o):
    e = o[5].split(':')[0]
    return int(e) if e not in ('0', '') and (int(o[3]) & 128) else None


def set_tap(o):
    """LN → tap：type 清 hold 位置 tap 位、endTime 清零。"""
    o[3] = str((int(o[3]) & ~128) | 1)
    ex = o[5].split(':')
    ex[0] = '0'
    o[5] = ':'.join(ex)


def polish(objs, rice=False, log=print):
    """输出修复：LN 钳制 + 纯米转换 + 可玩性合规。"""
    n0 = len(objs)
    # 0) 同刻同列去重
    seen, dedup = set(), 0
    keep = []
    for o in objs:
        k = (int(o[2]), col(o))
        if k in seen:
            dedup += 1
            continue
        seen.add(k)
        keep.append(o)
    objs = keep

    # 1) RICE：全部 LN → tap
    if rice:
        r = sum(1 for o in objs if ln_end(o) is not None)
        for o in objs:
            if ln_end(o) is not None:
                set_tap(o)
        log(f'  纯米转换: {r} 条 LN → tap')
    else:
        # 2) LN 时长钳制：<60ms 转 tap；>1 拍截断到 1/8 格线
        short = long_cap = 0
        for o in objs:
            e = ln_end(o)
            if e is None:
                continue
            st, t0 = int(o[2]), int(o[2])
            dur = e - st
            if dur < LN_MIN_MS:
                set_tap(o)
                short += 1
                continue
            if dur > LN_MAX_BEATS * BEAT:
                # 截断点：不超过 1 拍内的最近 1/8 格线（≥2 格）
                cap = st + LN_MAX_BEATS * BEAT
                grid = t0 + CELL * 2 * max(1, int((cap - t0) / (CELL * 2)))
                new_end = min(cap, grid)
                if new_end - st < LN_MIN_MS:
                    set_tap(o)
                    short += 1
                    continue
                ex = o[5].split(':')
                ex[0] = str(int(new_end))
                o[5] = ':'.join(ex)
                long_cap += 1
        log(f'  LN 钳制: 短条转 tap {short}  超长截断 {long_cap}')

        # 3) 局部 LN 墙压制：4s 滑窗 LN 行占比 >55% → 砍窗内最长的条
        #    （人类 LN 成段但窗内占比 ≤45%；v2 的「墙」就是窗内 80%+ 的重叠长条）
        rows = sorted({int(o[2]) for o in objs})
        row_has_ln = {}
        for o in objs:
            if ln_end(o) is not None:
                row_has_ln[int(o[2])] = True
        walls = 0
        i = 0
        while i < len(rows):
            j = i
            while j < len(rows) and rows[j] - rows[i] <= LN_WALL_WIN_MS:
                j += 1
            win = rows[i:j]
            ln_rows = [t for t in win if row_has_ln.get(t)]
            if len(win) >= 8 and len(ln_rows) / len(win) > LN_WALL_SHARE:
                excess = int(len(ln_rows) - LN_WALL_SHARE * len(win))
                # 该窗内最长的条优先砍
                cand = sorted(((ln_end(next(o for o in objs if int(o[2]) == t))
                                - t, t) for t in ln_rows), reverse=True)
                for _, t in cand[:excess]:
                    for o in objs:
                        if int(o[2]) == t and ln_end(o) is not None:
                            set_tap(o)
                            row_has_ln[t] = False
                            walls += 1
                            break
            i = j - 1 if j > i + 1 else i + 1
        log(f'  LN 墙压制: {walls} 条超密区长条转 tap')

        # 4) LN 期间同列 tap 删除（叠键）+ 释放后同列 <1 格的 LN 砍短
        del_set = set()
        for k, o in enumerate(objs):
            e = ln_end(o)
            if e is None:
                continue
            st, c = int(o[2]), col(o)
            for j, o2 in enumerate(objs):
                if col(o2) == c and st < int(o2[2]) < e:
                    del_set.add(j)
            # 释放后紧接同列音（<1 格）：尾提前 1 格
            nxt = min((int(o2[2]) for o2 in objs
                       if col(o2) == c and int(o2[2]) > e), default=None)
            if nxt is not None and nxt - e < CELL * 0.9:
                new_end = max(st + LN_MIN_MS, nxt - CELL)
                if new_end - st < LN_MIN_MS:
                    set_tap(o)
                else:
                    ex = o[5].split(':')
                    ex[0] = str(int(new_end))
                    o[5] = ':'.join(ex)
        if del_set:
            objs = [o for j, o in enumerate(objs) if j not in del_set]
        log(f'  叠键清理: 删 {len(del_set)}  释放间隔收紧')

    ln = sum(1 for o in objs if ln_end(o) is not None)
    log(f'  结果: {n0} → {len(objs)} 音符  LN {ln} ({ln/len(objs)*100:.1f}%)')
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mel', required=True, help='mel npz（build-audio-features 同款 [128,4096]）')
    ap.add_argument('--style', required=True, choices=sorted(STYLES))
    ap.add_argument('--ett', type=float, default=None, help='覆盖难度 token（5-35）')
    ap.add_argument('--version', required=True, help='谱面版本名，如 "AI v3"')
    ap.add_argument('--title', default='INTERNET YAMERO')
    ap.add_argument('--artist', default='KOTOKO')
    ap.add_argument('--creator', default='mug-prototype')
    ap.add_argument('--audio-name', default='INTERNET YAMERO.ogg')
    ap.add_argument('--vae', default='out/vae/vae.pt')
    ap.add_argument('--ckpt', default='out/diffusion/diff.pt')
    ap.add_argument('-o', required=True, help='输出 .osu 路径')
    args = ap.parse_args()

    st = STYLES[args.style]
    tok = dict(st['tok'])
    if args.ett is not None:
        tok['ett'] = args.ett

    vae = td.VAE()
    vae.load_state_dict(torch.load(args.vae, map_location='cpu'))
    vae.eval()
    vocab_size = max(max(v.values()) for v in td.VOCAB.values()) + 1
    model = td.Mug(vocab_size)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model.eval()

    mel = np.load(args.mel)['mel'].astype(np.float32)
    mel_t = torch.from_numpy(mel).unsqueeze(0)
    ctrl = torch.tensor([td.feature_to_ids(tok)], dtype=torch.long)
    uncond = torch.tensor([td.feature_to_ids({})], dtype=torch.long)
    print(f'风格 {args.style}（{st["cls"]}） token: {tok}')

    z = td.sample(model, mel_t, ctrl, scale=3.0, steps=50, uncond_ctrl=uncond)
    tmp = args.o + '.raw'
    n = td.sample_to_osu(vae, z, tmp, bpm=BPM)
    print(f'模型输出: {n} 音符（raw）')

    head, objs = parse_osu(tmp)
    objs = polish(objs, rice=(st['cls'] == 'rice'))

    # 元数据
    head = re.sub(r'AudioFilename: .*', f'AudioFilename: {args.audio_name}', head)
    head = re.sub(r'Title: .*', f'Title: {args.title}', head)
    head = re.sub(r'Artist: .*', f'Artist: {args.artist}', head)
    head = re.sub(r'Version: .*', f'Version: {args.version}', head)
    head = re.sub(r'Creator: .*', f'Creator: {args.creator}', head)
    head = re.sub(r'Tags: .*', f'Tags: ai mug-prototype {args.style}', head, flags=re.I)

    os.makedirs(os.path.dirname(args.o) or '.', exist_ok=True)
    write_osu(args.o, head, objs)
    os.remove(tmp)
    print('写出:', args.o)


if __name__ == '__main__':
    main()
