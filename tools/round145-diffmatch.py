#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充诊断（不是验收口径）：难度条件 --diff 与「原始占位率」对齐后，再看 p132。

验收口径固定 --diff 0.5 + target-occ 密度对齐。但当模型的**原始**占位率（P>0.5）
远低于数据集占位率时，靠「调低阈值」达标密度会把低置信格灌进来，p132 被压低。
本脚本扫几个 --diff，报告每个模型的原始占位率与对齐后 p132，用来判断
「1/32 成簇能力」是否真的没学出来，还是被密度对齐协议放大了。
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
def lm(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

STYLES = ['tech', 'jack', 'speed', 'stamina']
OCC = [0.1924, 0.2259, 0.1939, 0.1780]
DS = [0.5103, 0.0608, 0.6139, 0.1160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='out/diffusion/round145-perstyle')
    ap.add_argument('--diffs', default='0.5,0.65,0.8')
    ap.add_argument('--num', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    ev = lm('ev', os.path.join(HERE, 'eval-perstyle.py'))
    stats = lm('r141', os.path.join(HERE, 'round141-burst-stats.py'))
    res = {}
    for i, s in enumerate(STYLES):
        fp = os.path.join(a.root, f'{s}.pt')
        if not os.path.exists(fp):
            continue
        model, diff, pred, ck, _ = ev.load_model(fp, a.threads)
        res[s] = {}
        for dv in [float(x) for x in a.diffs.split(',')]:
            raw = ev.ddpm_sample(model, diff, a.num, dv, 'x0', a.seed).numpy()
            raw_occ = float((raw > 0.5).mean())
            thr = float(np.quantile(raw, 1.0 - OCC[i]))
            st = stats.pooled_stats((raw > thr).astype(np.uint8), raw)
            res[s][str(dv)] = {'原始占位率@0.5': round(raw_occ, 4),
                               'p132对齐后': st['p132'], '占位率对齐后': st['占位率'],
                               '阈值': round(thr, 4), '数据集占位率': OCC[i],
                               '数据集p132': DS[i],
                               'Δp132': round(st['p132'] - DS[i], 4)}
            print(f'{s:8s} diff={dv}: 原始占位率 {raw_occ:.4f}（数据 {OCC[i]:.4f}）'
                  f'  p132 {st["p132"]:.4f}（数据 {DS[i]:.4f}，Δ {st["p132"]-DS[i]:+.4f}）', flush=True)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f'→ {a.out}')


if __name__ == '__main__':
    main()
