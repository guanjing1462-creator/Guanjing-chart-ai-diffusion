#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""round 153 报告器：把 collect-*.json / impulse.json / train-*.jsonl 转成 markdown 表。

不做任何新计算，只做排版（数字全部来自评测脚本写出的 JSON），避免手抄出错。
用法：python3 tools/report-align.py [--collect out/diffusion/round153-align/collect-val.json]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIFTS = (-4, -1, 1, 4)


def j(path):
    with open(os.path.join(ROOT, path), encoding='utf-8') as f:
        return json.load(f)


def jl(path):
    rows = []
    with open(os.path.join(ROOT, path), encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def f3(v):
    return '—' if v is None else f'{v:.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect', default='out/diffusion/round153-align/collect-val.json')
    ap.add_argument('--collect-test', default='')
    ap.add_argument('--impulse', default='out/diffusion/round153-align/impulse.json')
    ap.add_argument('--s1-log', default='out/diffusion/round153-align/train-S1.jsonl')
    ap.add_argument('--s4-log', default='out/diffusion/round153-align/train-S4.jsonl')
    a = ap.parse_args()

    c = j(a.collect)
    L1 = c['config']['label_s1']
    L4 = c['config']['label_s4']
    S = c['summary']
    human = '人类（同 split 真谱）'

    print('### 训练曲线（逐 epoch val）\n')
    s1 = jl(a.s1_log)
    s4 = jl(a.s4_log)
    print('| epoch | %s val | %s val |' % (L1, L4))
    print('| --- | --- | --- |')
    for r1, r4 in zip(s1, s4):
        print(f'| {r1["epoch"]} | {r1["val"]:.4f} | {r4["val"]:.4f} |')
    print(f'| **最优** | **{min(r["val"] for r in s1):.4f}** | '
          f'**{min(r["val"] for r in s4):.4f}** |\n')

    print('### 主表（val，%d 窗）\n' % c['config']['num_windows'])
    print('| 组 | r250 | r1000 | p_odd_hi | 占用 | 音符 |')
    print('| --- | --- | --- | --- | --- | --- |')
    order = [human]
    for tag in (L1, L4):
        order += [f'{tag}-配对', f'{tag}-错配·对真C', f'{tag}-错配·对其给定C',
                  f'{tag}-零条件', f'{tag}-配对·占用校准', f'{tag}-错配·占用校准',
                  f'{tag}-配对·事后错配C']
    for tag in order:
        s = S[tag]
        print(f'| {"**"+tag+"**" if "配对" in tag and "事后" not in tag else tag} | '
              f'{f3(s["r250_中位"])} | {f3(s["r1000_中位"])} | '
              f'{f3(s["p_odd_hi_池化"])} | {s["占用"]:.3f} | {s["音符总数"]} |')
    print(f'\n结构零假设（均匀音符）p_odd_hi = {c["config"]["null_p_odd_hi_uniform"]:.3f}\n')

    print('### 条件平移（1 格 ≈ 40ms，4 格 ≈ 160ms）\n')
    print('| 组 | r250 对给定C | 对给定C/配对 | r250 对真C | p_odd_hi(对真C) | 输出位移估计 |')
    print('| --- | --- | --- | --- | --- | --- |')
    for tag in (L1, L4):
        base = S[f'{tag}-配对']['r250_中位']
        for k in SHIFTS:
            g = S[f'{tag}-条件{k:+d}格·对其给定C']
            t = S[f'{tag}-条件{k:+d}格·对真C']
            ratio = g['r250_中位'] / base if base else None
            print(f'| {tag} {k:+d} 格 | {f3(g["r250_中位"])} | {abs(ratio):.2f} | '
                  f'{f3(t["r250_中位"])} | {f3(t["p_odd_hi_池化"])} | '
                  f'{t["峰值位移_格"]:+d} 格 |')
        print(f'| **{tag} 配对（基准）** | **{f3(base)}** | 1.00 | '
              f'{f3(S[f"{tag}-配对"]["r250_中位"])} | '
              f'{f3(S[f"{tag}-配对"]["p_odd_hi_池化"])} | '
              f'{S[f"{tag}-配对"]["峰值位移_格"]:+d} 格 |')
    print()

    print('### r250 格移位扫描（同一批生成音符，把条件平移 k 格后算 r）\n')
    print('| 组 | ' + ' | '.join(f'{k:+d}' for k in (-8, -4, -2, -1, 0, 1, 2, 4, 8)) + ' |')
    print('| --- |' + ' --- |' * 9)
    for tag in (human, f'{L1}-配对', f'{L4}-配对'):
        sc = S[tag]['r250_格移位扫描中位']
        print(f'| {tag} | ' + ' | '.join(f3(sc[str(k)]) for k in (-8, -4, -2, -1, 0, 1, 2, 4, 8))
              + ' |')
    print()

    print('### 配对比较（逐窗差 / 胜-负 / Wilcoxon p）\n')
    print('| 对比 | r250 | r1000 |')
    print('| --- | --- | --- |')
    seen = []
    for key, v in c['compare'].items():
        if not v:
            continue
        name = key.split(' @')[0]
        if name not in seen:
            seen.append(name)
    for name in seen:
        cells = []
        for key in ('r250', 'r1000'):
            v = c['compare'].get(f'{name} @{key}')
            if not v:
                cells.append('—')
                continue
            p = v.get('wilcoxon_p')
            cells.append(f'{v["diff_mean"]:+.4f}（{v["wins"]}胜{v["losses"]}负'
                         + (f'，p={p:.3g}' if p is not None else '') + '）')
        print(f'| {name} | ' + ' | '.join(cells) + ' |')
    print()

    print('### p_odd_hi（1/32 踩格）：池化 + 逐窗 + bootstrap\n')
    print('| 对比 | 池化 | 差 | 逐窗 | bootstrap 95% CI | 单侧 p |')
    print('| --- | --- | --- | --- | --- | --- |')
    for k, v in c['p_odd_hi'].items():
        p, b = v['paired'], v['bootstrap']
        if not p:
            continue
        print(f'| {k} | {p["pooled_x"]:.4f} vs {p["pooled_y"]:.4f} | {p["pooled_diff"]:+.4f} | '
              f'{p["diff_mean"]:+.4f}（{p["wins"]}胜{p["losses"]}负'
              + (f'，p={p["wilcoxon_p"]:.3g}' if 'wilcoxon_p' in p else '') + '） | '
              f'[{b["ci95"][0]:+.4f}, {b["ci95"][1]:+.4f}] | {b["p_one_sided"]:.4f} |')
    print()

    print('### 条件因果效应（固定噪声，只改条件）\n')
    print('| 干预 | 平均\\|ΔP\\| | 二值翻转率 | 占用差 |')
    print('| --- | --- | --- | --- |')
    for k, v in c['cond_effect'].items():
        print(f'| {k} | {v["平均|ΔP|"]:.4f} | {v["二值翻转率"]:.3f} | {v["占用差"]:+.4f} |')
    print()

    ip = os.path.join(ROOT, a.impulse)
    if os.path.exists(ip):
        imp = j(a.impulse)['result']
        print('### 结构诊断：单格条件脉冲的输出响应（半高宽）\n')
        print('| 权重 | 峰值 \\|Δx0\\| | 半高宽（格） | ±1 相对 | ±2 相对 | ±4 相对 |')
        print('| --- | --- | --- | --- | --- | --- |')
        for tag, v in imp.items():
            print(f'| {tag} | {v["峰值|Δx0|"]:.5f} | {v["半高宽_格"]} | '
                  f'{v["offset±1_相对"]:.2f} | {v["offset±2_相对"]:.2f} | {v["offset±4_相对"]:.2f} |')
        print()
        print('归一化响应曲线（offset: 相对幅度）\n')
        for tag, v in imp.items():
            print(f'- {tag}: ' + ' '.join(f'{o:+d}:{x:.2f}'
                                          for o, x in v['归一化响应'].items()))
        print()

    if a.collect_test:
        ct = j(a.collect_test)
        St = ct['summary']
        print('### 独立复现（test）\n')
        print('| 组 | r250 | r1000 | p_odd_hi |')
        print('| --- | --- | --- | --- |')
        for tag in [human, f'{L1}-配对', f'{L1}-错配·对真C', f'{L1}-零条件',
                    f'{L4}-配对', f'{L4}-错配·对真C', f'{L4}-零条件']:
            s = St[tag]
            print(f'| {tag} | {f3(s["r250_中位"])} | {f3(s["r1000_中位"])} | '
                  f'{f3(s["p_odd_hi_池化"])} |')
        print()
        for k, v in ct['p_odd_hi'].items():
            p, b = v['paired'], v['bootstrap']
            if not p:
                continue
            print(f'- {k}: 池化 {p["pooled_x"]:.4f} vs {p["pooled_y"]:.4f}'
                  f'（差 {p["pooled_diff"]:+.4f}；bootstrap 95% CI '
                  f'[{b["ci95"][0]:+.4f}, {b["ci95"][1]:+.4f}]，单侧 p={b["p_one_sided"]:.4f}）')
        print()


if __name__ == '__main__':
    main()
