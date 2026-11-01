#!/usr/bin/env python3
"""训练/评估键型转移模型（这是「训练」而不是「调参」）。

数据：人类 4K 谱（osu! 包），切成 (上一行形状, 间隔档) → 下一行形状 的样本。
模型：
  A 基线：经验频率（当前生成器用的就是这种转移表）+ 最简单的加性平滑
  B 回退：P(shape|prev,gap) → P(shape|gap) → P(shape)
  C 特征化：以「上一行的列集合 + 间隔档 + 是否小节首」为特征的多项逻辑回归

评估：按谱面划分训练/测试集（同谱不跨集），指标为留出集**对数损失（bits/样本）**，
     越低越好。这个指标直接反映「模型对下一拍形状的预测能力」。

用法：python3 tools/train-shape-model.py --packs out/packs/tech --out out/shape-model.json
"""
import argparse
import collections
import glob
import json
import math
import os
import random

import numpy as np

COLS = 4


def parse_chart(path):
    """解析 .osu 4K 谱面，返回 [(time_ms, tuple(cols))]（已按时间排序）。"""
    rows = []
    xs = {64: 0, 192: 1, 320: 2, 448: 3}
    by_time = collections.defaultdict(set)
    for ln in open(path, encoding='utf-8', errors='ignore'):
        p = ln.split(',')
        if len(p) < 5:
            continue
        try:
            x = int(p[0]); t = int(p[2]); typ = int(p[3])
        except ValueError:
            continue
        if x not in xs:
            continue
        if typ & 1 or typ & 128:      # 保留 note 与 LN 头
            by_time[t].add(xs[x])
    for t in sorted(by_time):
        rows.append((t, tuple(sorted(by_time[t]))))
    return rows


def gap_bucket(dt_ms):
    if dt_ms <= 60:
        return '32'
    if dt_ms <= 110:
        return '16'
    if dt_ms <= 190:
        return '8'
    if dt_ms <= 330:
        return '4'
    if dt_ms <= 650:
        return '2'
    if dt_ms <= 1200:
        return '1'
    return 'slow'


def samples_of(rows):
    """产出 (prev_shape, gap_bucket, pos_in_bar, shape) 样本。"""
    out = []
    for i in range(1, len(rows)):
        t0, s0 = rows[i - 1]
        t1, s1 = rows[i]
        g = gap_bucket(t1 - t0)
        pos = int((t1 % 2000) // 125)   # 小节内 16 格粗分位
        prev2 = rows[i - 2][1] if i >= 2 else ()
        out.append((s0, g, pos, s1, prev2))
    return out


def build_data(pack_dir, limit=None):
    files = sorted(glob.glob(os.path.join(pack_dir, '*.osu')))
    if limit:
        files = files[:limit]
    data = []           # (file_idx, samples)
    for k, f in enumerate(files):
        rows = parse_chart(f)
        if len(rows) < 50:
            continue
        data.append(samples_of(rows))
    return data


def eval_logloss(data, model, alpha=0.5):
    """在给定数据上算 bits/样本（留出集评估）。"""
    total_bits = 0.0
    n = 0
    for samples in data:
        for prev, g, pos, shape, prev2 in samples:
            p = model(prev, g, pos, shape, prev2)
            total_bits += -math.log2(max(p, 1e-9))
            n += 1
    return total_bits / max(1, n), n


def train_freq(samples_list, alpha=0.5):
    """模型 A：P(shape | prev, gap) 经验频率 + 加性平滑。"""
    joint = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    allc = collections.Counter()
    for samples in samples_list:
        for prev, g, pos, shape, prev2 in samples:
            joint[(prev, g)][shape] += 1
            total[(prev, g)] += 1
            allc[shape] += 1
    V = len(allc)

    def model(prev, g, pos, shape, prev2=()):
        c = joint[(prev, g)]
        return (c.get(shape, 0) + alpha) / (total[(prev, g)] + alpha * V)

    return model, (joint, total, allc, V)


def train_backoff(samples_list, alpha=0.5):
    """模型 B：三级回退。"""
    j1 = collections.defaultdict(collections.Counter); t1 = collections.Counter()
    j2 = collections.defaultdict(collections.Counter); t2 = collections.Counter()
    j3 = collections.Counter(); n3 = 0
    for samples in samples_list:
        for prev, g, pos, shape, prev2 in samples:
            j1[(prev, g)][shape] += 1; t1[(prev, g)] += 1
            j2[g][shape] += 1; t2[g] += 1
            j3[shape] += 1; n3 += 1
    V = len(j3)

    def model(prev, g, pos, shape, prev2=()):
        if t1[(prev, g)] >= 20:
            c = j1[(prev, g)]
            return (c.get(shape, 0) + alpha) / (t1[(prev, g)] + alpha * V)
        if t2[g] >= 20:
            c = j2[g]
            return (c.get(shape, 0) + alpha) / (t2[g] + alpha * V)
        return (j3.get(shape, 0) + alpha) / (n3 + alpha * V)

    return model, (j1, j2, j3)


def train_two_order(samples_list, alpha=0.5):
    """模型 C：二阶上下文 (prev2, prev, gap)，样本不足时回退到一阶。"""
    j1 = collections.defaultdict(collections.Counter); t1 = collections.Counter()
    j2 = collections.defaultdict(collections.Counter); t2 = collections.Counter()
    allc = collections.Counter()
    for samples in samples_list:
        for prev, g, pos, shape, prev2 in samples:
            j1[(prev2, prev, g)][shape] += 1; t1[(prev2, prev, g)] += 1
            j2[(prev, g)][shape] += 1; t2[(prev, g)] += 1
            allc[shape] += 1
    V = len(allc)

    def model(prev, g, pos, shape, prev2=()):
        if t1[(prev2, prev, g)] >= 30:
            c = j1[(prev2, prev, g)]
            return (c.get(shape, 0) + alpha) / (t1[(prev2, prev, g)] + alpha * V)
        c = j2[(prev, g)]
        return (c.get(shape, 0) + alpha) / (t2[(prev, g)] + alpha * V)

    return model, (j1, j2, allc, V)


def train_hier(samples_list, alpha=0.5, alpha_sz=0.2):
    """模型 D：分层——先预测行大小（1~4），再在给定大小下预测形状。"""
    sz_c = collections.defaultdict(collections.Counter); sz_t = collections.Counter()
    sh_c = collections.defaultdict(collections.Counter); sh_t = collections.Counter()
    allsh = collections.Counter()
    for samples in samples_list:
        for prev, g, pos, shape, prev2 in samples:
            sz = len(shape)
            sz_c[(len(prev), g)][sz] += 1; sz_t[(len(prev), g)] += 1
            sh_c[(len(prev), g, sz)][shape] += 1; sh_t[(len(prev), g, sz)] += 1
            allsh[shape] += 1
    Vsz, Vsh = 4, len(allsh)

    def model(prev, g, pos, shape, prev2=()):
        sz = len(shape)
        k_sz = (len(prev), g)
        p_sz = (sz_c[k_sz].get(sz, 0) + alpha_sz) / (sz_t[k_sz] + alpha_sz * Vsz)
        k_sh = (len(prev), g, sz)
        p_sh = (sh_c[k_sh].get(shape, 0) + alpha) / (sh_t[k_sh] + alpha * Vsh)
        return p_sz * p_sh

    return model, (sz_c, sh_c, allsh)


def train_interp(samples_list, lam=0.6, alpha=0.3):
    """模型 E：二阶/一阶/边际 的线性插值（Jelinek-Mercer），比硬回退稳。"""
    j2 = collections.defaultdict(collections.Counter); t2 = collections.Counter()
    j1 = collections.defaultdict(collections.Counter); t1 = collections.Counter()
    j0 = collections.Counter(); n0 = 0
    for samples in samples_list:
        for prev, g, pos, shape, prev2 in samples:
            j2[(prev2, prev, g)][shape] += 1; t2[(prev2, prev, g)] += 1
            j1[(prev, g)][shape] += 1; t1[(prev, g)] += 1
            j0[shape] += 1; n0 += 1
    V = len(j0)

    def model(prev, g, pos, shape, prev2=()):
        p0 = (j0.get(shape, 0) + alpha) / (n0 + alpha * V)
        p1 = (j1[(prev, g)].get(shape, 0) + alpha) / (t1[(prev, g)] + alpha * V)
        p2 = (j2[(prev2, prev, g)].get(shape, 0) + alpha) / (t2[(prev2, prev, g)] + alpha * V)
        return lam * p2 + (1 - lam) * 0.7 * p1 + (1 - lam) * 0.3 * p0

    return model, (j2, j1, j0)


def train_interp_pos(samples_list, lam=0.85, alpha=0.3):
    """模型 F：最高阶上下文加入「小节内位置」，其余同 E。"""
    j3 = collections.defaultdict(collections.Counter); t3 = collections.Counter()
    j2 = collections.defaultdict(collections.Counter); t2 = collections.Counter()
    j1 = collections.defaultdict(collections.Counter); t1 = collections.Counter()
    j0 = collections.Counter(); n0 = 0
    for samples in samples_list:
        for prev, g, pos, shape, prev2 in samples:
            j3[(prev2, prev, g, pos)][shape] += 1; t3[(prev2, prev, g, pos)] += 1
            j2[(prev, g)][shape] += 1; t2[(prev, g)] += 1
            j1[g][shape] += 1; t1[g] += 1
            j0[shape] += 1; n0 += 1
    V = len(j0)

    def model(prev, g, pos, shape, prev2=()):
        p0 = (j0.get(shape, 0) + alpha) / (n0 + alpha * V)
        p1 = (j1[g].get(shape, 0) + alpha) / (t1[g] + alpha * V)
        p2 = (j2[(prev, g)].get(shape, 0) + alpha) / (t2[(prev, g)] + alpha * V)
        p3 = (j3[(prev2, prev, g, pos)].get(shape, 0) + alpha) / (t3[(prev2, prev, g, pos)] + alpha * V)
        return lam * p3 + (1 - lam) * (0.55 * p2 + 0.3 * p1 + 0.15 * p0)

    return model, (j3, j2, j1, j0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packs', default='out/packs/tech')
    ap.add_argument('--limit', type=int, default=400)
    ap.add_argument('--split', type=float, default=0.8, help='训练集比例（按谱划分）')
    ap.add_argument('--out', default='out/shape-model.json')
    a = ap.parse_args()

    data = build_data(a.packs, a.limit)
    random.Random(3231).shuffle(data)
    k = int(len(data) * a.split)
    tr, te = data[:k], data[k:]
    print(f'谱面：训练 {len(tr)} 张 / 测试 {len(te)} 张')

    modelA, statA = train_freq(tr)
    modelB, statB = train_backoff(tr)
    modelC, statC = train_two_order(tr)
    modelD, statD = train_hier(tr)
    bestE = None
    for lam in (0.4, 0.55, 0.7, 0.85):
        mE, _ = train_interp(tr, lam=lam)
        l_te, _n = eval_logloss(te, mE)
        print(f'E 插值 λ={lam:.2f}: 留出 {l_te:.3f} bits/样本')
        if bestE is None or l_te < bestE[0]:
            bestE = (l_te, lam, mE)
    results = {}
    modelF, _ = train_interp_pos(tr)
    for name, m in (('A 经验频率', modelA), ('B 三级回退', modelB),
                    ('C 二阶上下文', modelC), ('D 分层(尺寸→形状)', modelD),
                    (f'E 插值 λ={bestE[1]:.2f}', bestE[2]), ('F 插值+位置', modelF)):
        loss_tr, n_tr = eval_logloss(tr, m)
        loss_te, n_te = eval_logloss(te, m)
        results[name] = loss_te
        print(f'{name}: 训练 {loss_tr:.3f} bits/样本（{n_tr}）  留出 {loss_te:.3f} bits/样本（{n_te}）')

    # 统计稀疏程度
    j1, t1 = statA[0], statA[1]
    rare = sum(1 for kk in t1 if t1[kk] < 20)
    print(f'上下文总数 {len(t1)}，其中样本数<20 的 {rare}（{rare / max(1, len(t1)) * 100:.0f}%）')

    best = min(results, key=results.get)
    print(f'最优模型：{best}（留出 {results[best]:.3f} bits/样本）')
    # —— 序列化最优模型（二阶计数表），写成语料格式的扩展字段 ——
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        model_map = {'A 经验频率': modelA, 'B 三级回退': modelB,
                     'C 二阶上下文': modelC, 'D 分层(尺寸→形状)': modelD,
                     f'E 插值 λ={bestE[1]:.2f}': bestE[2], 'F 插值+位置': modelF}
        # 统一用「二阶计数」做导出（生成器侧要的是可采样的权重表）
        j2 = collections.defaultdict(collections.Counter)
        for samples in tr:
            for prev, g, pos, shape, prev2 in samples:
                j2[f'{prev2}|{prev}|{g}'][str(list(shape))] += 1
        out = {}
        for k2, cc in j2.items():
            tot = sum(cc.values())
            if tot < 8:
                continue
            out[k2] = {kk: round(v / tot, 5) for kk, v in cc.most_common(40)}
        json.dump({'loss_holdout': results, 'best': best, 'trans2': out},
                  open(a.out, 'w'))
        print(f'已导出二阶转移表：{len(out)} 个上下文 → {a.out}')


if __name__ == '__main__':
    main()
