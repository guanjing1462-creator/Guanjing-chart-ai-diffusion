#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格灵敏度探针（不采样，只前向）：同一份噪声/同一个 t/同一个难度条件，只改 style。

为什么需要它：5 窗采样的 p132 有 ±0.02 的抽样噪声（四风格 × 5 窗），而「风格条件到底有没有
把权重用起来」这件事可以在解析层面直接量：固定 xt，把 style 从 0 换到 3，看 x0 预测怎么变。

指标（与 tools/train-diffusion32-film.py 里的 --style-probe 同一套口径）：
  cross_style_mae : 4 风格两两之间 x0 预测的平均绝对差（6 对再平均）
  noise_mae       : 同风格、换一份噪声时的平均绝对差（同一 t 下「内容不同」的尺度）
  style/noise     : 比值；越接近 0 = 风格对输出的影响越像噪声级，条件越弱

用法：
  python3 tools/film-style-probe.py --ckpt <ckpt> [--ckpt <ckpt2>] --out out/.../probe.json
"""
import argparse
import importlib.util
import json
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
STYLES = ['tech', 'jack', 'speed', 'stamina']


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe(ckpt, data, t_val=100, n=1024, seed=4242, threads=1):
    S = load_module('occ_sampler', os.path.join(HERE, 'sample-diffusion32film.py'))
    d = np.load(data)
    split = np.asarray(d['split'])
    va = np.where(split == 1)[0]
    X = np.asarray(d['X'])[va]
    occ = np.asarray(d['X']).reshape(len(d['X']), -1).mean(1).astype(np.float32)
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    dc = ck.get('difficulty_condition') or {}
    lo, hi = float(dc.get('lo', 0.0)), float(dc.get('hi', 1.0))
    cont = np.clip((occ - lo) / max(1e-6, hi - lo), 0, 1).astype(np.float32)[va]
    model, diff, info = S.load_model(ckpt, False, 48, 200, threads, 0,
                                     pred=ck.get('pred'))
    model.eval()
    W = int(X.shape[2])
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)[:min(n, len(X))].numpy()
    x0 = torch.from_numpy(np.ascontiguousarray(X[idx])).float()
    dd = torch.from_numpy(cont[idx])
    tt = torch.full((len(idx),), int(t_val), dtype=torch.long)
    no = torch.randn(len(idx), 4, W, generator=g)
    no2 = torch.randn(len(idx), 4, W, generator=g)
    xt, xt2 = diff.q(x0, tt, no), diff.q(x0, tt, no2)
    with torch.no_grad():
        outs = [model(xt, tt, torch.full((len(idx),), s, dtype=torch.long), dd, None)
                for s in range(4)]
        second = model(xt2, tt, torch.zeros(len(idx), dtype=torch.long), dd, None)
    pairs, vals = {}, []
    for i in range(4):
        for j in range(i + 1, 4):
            v = float((outs[i] - outs[j]).abs().mean())
            pairs[f'{STYLES[i]}-{STYLES[j]}'] = round(v, 6)
            vals.append(v)
    cross = float(np.mean(vals))
    noise = float((outs[0] - second).abs().mean())
    res = {'ckpt': ckpt, 'style_cond': info.get('style_cond'), 'val': ck.get('val'),
           'epoch': ck.get('epoch'), 'params': ck.get('params'),
           'params_style_path': ck.get('params_style_path'),
           't': t_val, 'n_windows': int(len(idx)), 'seed': seed,
           'pred': ck.get('pred'),
           'cross_style_mae': round(cross, 6), 'cross_style_pairs': pairs,
           'same_style_other_noise_mae': round(noise, 6),
           'style_over_noise': round(cross / max(1e-12, noise), 4),
           'x0_std': round(float(x0.std()), 6),
           'cross_style_over_x0std': round(cross / max(1e-12, float(x0.std())), 4)}
    # ---- 换错 style 的损失代价（更直接：模型到底依赖不依赖 style 这个标签）----
    g2 = torch.Generator().manual_seed(1234)
    vt = torch.randint(0, 200, (len(X),), generator=g2)
    vn = torch.randn(len(X), 4, W, generator=g2)
    Xv = torch.from_numpy(np.ascontiguousarray(X)).float()
    Dv = torch.from_numpy(cont)
    Sv = torch.from_numpy(np.asarray(d['style'])[va].astype(np.int64))
    loss_by = torch.zeros(4, len(Xv))
    with torch.no_grad():
        for i in range(0, len(Xv), 512):
            sl = slice(i, min(i + 512, len(Xv)))
            xtv = diff.q(Xv[sl], vt[sl], vn[sl])
            # 采样器里的 Diffusion 没有 target（训练侧才有），这里按 pred 直接反推目标
            a_t = diff.abar[vt[sl]][:, None, None]
            mode = ck.get('pred', 'x0')
            if mode == 'x0':
                tgt = Xv[sl]
            elif mode == 'eps':
                tgt = vn[sl]
            else:
                tgt = a_t.sqrt() * vn[sl] - (1 - a_t).sqrt() * Xv[sl]
            for s_ in range(4):
                pv = model(xtv, vt[sl], torch.full((sl.stop - sl.start,), s_, dtype=torch.long), Dv[sl], None)
                loss_by[s_, sl] = ((pv - tgt) ** 2).mean(dim=(1, 2))
    own = loss_by.gather(0, Sv[None]).squeeze(0)
    mask = torch.ones(4, len(Sv)) - torch.nn.functional.one_hot(Sv, 4).T.float()
    others = (loss_by * mask).sum(0) / mask.sum(0).clamp(min=1)
    res['swap_style'] = {
        'own_style_loss': round(float(own.mean()), 6),
        'wrong_style_loss': round(float(others.mean()), 6),
        'delta': round(float(others.mean() - own.mean()), 6),
        'delta_rel': round(float((others.mean() - own.mean()) / own.mean()), 4),
        'per_style_delta': {STYLES[s_]: round(float(others[Sv == s_].mean() - own[Sv == s_].mean()), 6)
                            for s_ in range(4)},
        'note': '同一份固定 t/噪声（seed 1234）下，用错 style 标签的 val 损失 - 用对标签的损失；'
                '差值越接近 0，说明模型几乎不用 style'}
    print(f'  换错 style 的代价: own {res["swap_style"]["own_style_loss"]:.5f} → '
          f'wrong {res["swap_style"]["wrong_style_loss"]:.5f} '
          f'(Δ {res["swap_style"]["delta"]:+.5f}, {100*res["swap_style"]["delta_rel"]:+.2f}%)', flush=True)
    if 'style_probe' in ck:
        res['ckpt_style_probe'] = ck['style_probe']
    if 'val_per_style' in ck:
        res['val_per_style'] = ck['val_per_style']
    print(f'{os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}  '
          f'[{info.get("style_cond")}]  cross_style_mae {cross:.4f}  '
          f'noise_mae {noise:.4f}  style/noise {cross / max(1e-12, noise):.4f}  '
          f'(x0 std {float(x0.std()):.4f})', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--data', default='out/diff-data-grouped/dataset.npz')
    ap.add_argument('--t', type=int, default=100)
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    res = {}
    for c in a.ckpt:
        res[c] = probe(c, a.data, t_val=a.t, n=a.n, threads=a.threads)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
