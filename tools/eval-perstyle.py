#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按风格分模型的验收采样器 / 统计器（第 145 轮）。

与 tools/sample-diffusion32occ.py 的区别：**模型里没有 style 条件**
（`unet1d-perstyle-fourier-v1`，模型身份 = 风格），所以：
  - 不再需要 `--style` 去控制输出，只按 checkpoint 里记录的 `only_style` 采样；
  - 仍然保留 `--target-occ`（按该风格数据集占位率反解阈值，密度对齐后比 p132）；
  - 连续难度条件 `--diff` 照旧吃该模型自己 train 子集归一化的 [0,1] 标量。

模型类**从训练脚本 import**（不是复制），保证结构与训练逐字段一致；
不是同一份权重就直接 strict 加载失败报错，绝不静默跳过参数。

统计口径：直接复用 tools/round141-burst-stats.py 的 pooled_stats
（p132 = 相邻占用行间隔 == 1 格的比例，跨窗口池化），与第 141/143/144 轮报表同口径。

用法：
  python3 tools/eval-perstyle.py --root out/diffusion/round145-perstyle \
      --out out/diffusion/round145-perstyle/eval.json
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
STYLES = ['tech', 'jack', 'speed', 'stamina']
# 数据集（train split==0，本风格）占位率 —— 密度对齐目标（与 141/143/144 轮同一套数）
OCC = [0.1924, 0.2259, 0.1939, 0.1780]
# 数据集（train split==0，本风格）p132 —— 判定基准
DATASET_P132 = [0.5103, 0.0608, 0.6139, 0.1160]
WIN = 256
NLANES = 4


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prediction_to_x0(xt, out, abar, pred):
    if pred == 'x0':
        return out
    if pred == 'v':
        return abar.sqrt() * xt - (1 - abar).sqrt() * out
    if pred == 'eps':
        return (xt - (1 - abar).sqrt() * out) / abar.sqrt()
    raise ValueError(f'未知预测目标 {pred!r}')


@torch.no_grad()
def ddpm_sample(model, diff, n, diff_value, pred, seed):
    """标准 DDPM 祖先采样：x_T ~ N(0,I) → x_0。返回 [n,4,256] 连续值（近似 x0）。"""
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(n, NLANES, WIN, generator=g)
    d = torch.full((n,), float(diff_value), dtype=torch.float32)
    T = diff.T
    for t in reversed(range(T)):
        tb = torch.full((n,), t, dtype=torch.long)
        out = model(x, tb, d)                       # 无 style 张量
        ab, a_t = diff.abar[t], diff.alphas[t]
        x0 = prediction_to_x0(x, out, ab, pred)
        if pred == 'x0':
            x0 = x0.clamp(-1.0, 2.0)
        if t > 0:
            ab_prev = diff.abar[t - 1]
            mean = (ab_prev.sqrt() * diff.betas[t] * x0
                    + a_t.sqrt() * (1 - ab_prev) * x) / (1 - ab)
            sigma = (diff.betas[t] * (1 - diff.abar[t - 1]) / (1 - ab)).sqrt()
            x = mean + sigma * torch.randn(n, NLANES, WIN, generator=g)
        else:
            x = x0
    return x


def load_model(ckpt, threads=1):
    """strict 加载 per-style 权重；结构/元数据不一致就报错，不猜。"""
    torch.set_num_threads(max(1, threads))
    ps = load_module('ps_train', os.path.join(HERE, 'train-diffusion32-perstyle.py'))
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict) or 'state' not in ck:
        raise SystemExit(f'[错误] {ckpt} 不是训练器写的 checkpoint')
    if ck.get('model_type') != 'unet1d-perstyle-fourier-v1':
        raise SystemExit(f'[错误] {ckpt} 的 model_type={ck.get("model_type")!r}，'
                         f'不是按风格分模型版（期望 unet1d-perstyle-fourier-v1）')
    state = ck['state']
    if 'freq' not in state:
        raise SystemExit(f'[错误] {ckpt} 缺 freq（连续难度条件的 Fourier 基底），不允许随机补齐')
    if 'c_style.weight' in state:
        raise SystemExit(f'[错误] {ckpt} 里还有 c_style（风格嵌入），不是本轮的按风格分模型权重')
    # Conv1d(cin, base, 3).weight 形状是 [base, cin, k]：第 0 维是 base，第 1 维才是 cin
    base = int(ck.get('base', state['inp.weight'].shape[0]))
    cin, temb_dim = state['inp.weight'].shape[1], state['freq'].shape[0] * 2
    if int(ck.get('base', base)) != state['inp.weight'].shape[0]:
        raise SystemExit(f'[错误] {ckpt} 的 base 元数据与权重形状不一致')
    cond_dim = int(ck.get('cond_dim', 0))
    if cin != NLANES or int(ck.get('rows', WIN)) != WIN or ck.get('grid', '1/32') != '1/32':
        raise SystemExit(f'[错误] {ckpt} 的窗口/网格元数据不适用于 1/32、{WIN} 格采样器')
    model = ps.UNet1D(cin=cin, base=base, temb_dim=temb_dim, cond_dim=cond_dim)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise SystemExit(f'[错误] {ckpt} strict 加载失败（结构与训练不一致）：{e}')
    model.eval()
    pred = ck.get('pred')
    if pred not in ('eps', 'x0', 'v'):
        raise SystemExit(f'[错误] {ckpt} 缺少合法 pred 元数据（{pred!r}）')
    nparam = sum(p.numel() for p in model.parameters())
    return model, ps.Diffusion(int(ck['timesteps'])), pred, ck, nparam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='out/diffusion/round145-perstyle')
    ap.add_argument('--styles', default='tech,jack,speed,stamina',
                    help='要评估的风格（缺失的 ckpt 会跳过并标注）')
    ap.add_argument('--num', type=int, default=5, help='每模型采样窗口数（任务书：5）')
    ap.add_argument('--diff', type=float, default=0.5, help='连续难度条件（四模型统一 0.5）')
    ap.add_argument('--seed', type=int, default=0, help='噪声种子（四模型相同 → 初始噪声逐元素一致）')
    ap.add_argument('--common-occ', type=float, default=0.1986,
                    help='附加验收：把所有模型都对齐到**同一个**占位率（全 train 0.1986），'
                         '排除「密度对齐本身造成差异」。0 = 不做')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default='')
    ap.add_argument('--outdir', default='', help='rolls npz 输出目录（默认 <root>/samples）')
    a = ap.parse_args()

    stats = load_module('r141stats', os.path.join(HERE, 'round141-burst-stats.py'))
    styles = [s.strip() for s in a.styles.split(',') if s.strip()]
    outdir = a.outdir or os.path.join(a.root, 'samples')
    os.makedirs(outdir, exist_ok=True)
    res = {'root': a.root, 'num': a.num, 'diff': a.diff, 'seed': a.seed,
           'common_occ': a.common_occ,
           'protocol': '每模型只采它自己的风格；5 窗；diff=0.5；按该风格数据集占位率反解阈值'
                       '（密度对齐）；p132 = 相邻占用行间隔==1 的比例，跨窗口池化'
                       '（round141-burst-stats.pooled_stats）',
           'dataset_p132': dict(zip(STYLES, DATASET_P132)),
           'dataset_occ': dict(zip(STYLES, OCC)),
           'models': {}}

    for name in styles:
        i = STYLES.index(name)
        ckpt = os.path.join(a.root, f'{name}.pt')
        if not os.path.exists(ckpt):
            print(f'[跳过] {name}: 没有 {ckpt}', flush=True)
            continue
        model, diff, pred, ck, nparam = load_model(ckpt, a.threads)
        if int(ck.get('only_style', -1)) != i:
            raise SystemExit(f'[错误] {ckpt} 记录的 only_style={ck.get("only_style")}，'
                             f'与文件名 {name}({i}) 不一致')
        m = {'ckpt': ckpt, 'pred': pred, 'params': nparam,
             'base': int(ck['base']), 'timesteps': int(ck['timesteps']),
             'val': ck.get('val'), 'epoch': ck.get('epoch'),
             'train_seconds': ck.get('train_seconds'),
             'train_windows': ck.get('train_windows'),
             'train_groups': ck.get('train_groups'),
             'val_windows': ck.get('val_windows'),
             'difficulty_condition': ck.get('difficulty_condition'),
             'style_condition': ck.get('style_condition')}
        runs = []
        # 1) 主验收：用该风格数据集占位率做密度对齐
        runs.append(('own', OCC[i]))
        # 2) 附加验收：四模型统一占位率（排除密度差异）
        if a.common_occ and abs(a.common_occ - OCC[i]) > 1e-6:
            runs.append(('common', a.common_occ))
        for tag, occ in runs:
            t0 = torch.get_num_threads()
            raw = ddpm_sample(model, diff, a.num, a.diff, pred, a.seed).numpy()
            thr = float(np.quantile(raw, 1.0 - occ))
            X = (raw > thr).astype(np.uint8)
            st = stats.pooled_stats(X, raw)
            st['目标占位率'] = occ
            st['阈值'] = round(thr, 6)
            st['数据集p132'] = DATASET_P132[i]
            st['p132减数据集'] = round(st['p132'] - DATASET_P132[i], 4)
            st['达标_0.05'] = bool(abs(st['p132'] - DATASET_P132[i]) <= 0.05)
            m[tag] = st
            fn = os.path.join(outdir, f'rolls-{name}-{tag}-s{a.seed}.npz')
            np.savez_compressed(fn, X=X, P=raw.astype(np.float32),
                                style=np.int64(i), diff=np.float64(a.diff),
                                pred=np.str_(pred), seed=np.int64(a.seed),
                                threshold=np.float64(thr), target_occ=np.float64(occ))
            print(f'[{name}] {tag}: 占位率 {st["占位率"]:.4f}（目标 {occ:.4f}）  '
                  f'p132 {st["p132"]:.4f}（数据集 {DATASET_P132[i]:.4f}，'
                  f'Δ {st["p132减数据集"]:+.4f}）  阈值 {thr:.3f}', flush=True)
        res['models'][name] = m

    # ---- 汇总表 ----
    print('\n=== 四风格 p132 三方对照（数据集 / 单模型基线 / 分模型）===')
    BASE = {'tech': 0.3962, 'jack': 0.0923, 'speed': 0.3562, 'stamina': 0.0208}
    print(f'{"style":>8} {"数据集":>7} {"单模型":>7} {"分模型":>7} {"Δ分模型":>8} {"达标":>5}')
    rows = {}
    for name in styles:
        if name not in res['models']:
            continue
        st = res['models'][name]['own']
        i = STYLES.index(name)
        rows[name] = {'dataset': DATASET_P132[i], 'single': BASE[name],
                      'perstyle': st['p132'],
                      'delta': round(st['p132'] - DATASET_P132[i], 4),
                      'ok': bool(abs(st['p132'] - DATASET_P132[i]) <= 0.05)}
        print(f'{name:>8} {DATASET_P132[i]:>7.4f} {BASE[name]:>7.4f} {st["p132"]:>7.4f} '
              f'{rows[name]["delta"]:>+8.4f} {"✅" if rows[name]["ok"] else "❌":>5}')
    res['table'] = rows
    if rows:
        res['ok_count'] = sum(1 for v in rows.values() if v['ok'])
        res['n_models'] = len(rows)
        res['max_abs_delta'] = round(max(abs(v['delta']) for v in rows.values()), 4)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
