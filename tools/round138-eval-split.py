#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round138 · 用训练好的权重在指定 split 上算损失（池化 / 组均 / 分风格）。

为什么单独一个脚本：训练器只在训练时评 val + A/B；test 只允许**最后跑一次**，
所以留在训练结束后单独触发，且不参与任何选型。

口径（与 tools/train-diffusion32-grouped.py 的 eval_loss 逐项一致）：
  - 固定 t 与噪声（generator seed 1234，按窗口数 n 生成，逐元素可复现）；
  - 未加权 MSE：以 x0（干净谱面）为目标，loss = (pred - x0)^2；
  - 连续难度条件用 ckpt 里记录的 lo/hi 复算占位率归一化（不在 val/test 上重拟合）。

用法：
  python3 tools/round138-eval-split.py --ckpt <权重> --split 2 \
      --out out/diffusion/round138-train/eval-test.json
"""
import argparse, importlib.util, json, os
import numpy as np
import torch
import torch.nn.functional as F


def load_trainer_module(path):
    spec = importlib.util.spec_from_file_location('t32g', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # 主逻辑在 main() 下，导入不会训练
    return mod


def eval_set(model, diff, pred, Xe, Se, De, lbl, timesteps, batch=512):
    n = len(Xe)
    g = torch.Generator().manual_seed(1234)
    vt = torch.randint(0, timesteps, (n,), generator=g)
    vn = torch.randn(n, 4, Xe.shape[2], generator=g)
    vec = torch.zeros(n)
    with torch.no_grad():
        for i in range(0, n, batch):
            m = min(batch, n - i)
            sl = slice(i, i + m)
            xt = diff.q(Xe[sl], vt[sl], vn[sl])
            p = model(xt, vt[sl], Se[sl], De[sl], None)
            _, tgt, _ = diff.target(Xe[sl], vt[sl], vn[sl], pred)
            vec[sl] = ((p - tgt) ** 2).mean(dim=(1, 2))
    pooled = float(vec.mean())
    gm, per_grp = None, {}
    if lbl is not None:
        k = int(lbl.max().item()) + 1
        per = torch.zeros(k).scatter_add_(0, lbl, vec)
        cnt = torch.zeros(k).scatter_add_(0, lbl, torch.ones_like(vec))
        gm = float((per / cnt.clamp(min=1.0)).mean())
        per_grp = {int(i): round(float(v), 6) for i, v in enumerate((per / cnt.clamp(min=1.0)))}
    return pooled, gm, vec.numpy(), per_grp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='out/diff-data-grouped/dataset.npz')
    ap.add_argument('--trainer', default='tools/train-diffusion32-grouped.py')
    ap.add_argument('--split', type=int, default=2, choices=[0, 1, 2])
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    mod = load_trainer_module(a.trainer)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    state = ck['state']
    base = int(ck['base'])
    T = int(ck.get('timesteps', 200))
    pred = ck['pred']
    dc = ck['difficulty_condition']
    print(f'权重 {a.ckpt}')
    print(f'  epoch {ck.get("epoch")} / pred {pred} / base {base} / T {T} / '
          f'难度条件 lo {dc["lo"]:.4f} hi {dc["hi"]:.4f}')
    print(f'  split_policy {json.dumps(ck.get("split_policy", {}), ensure_ascii=False)}')
    model = mod.UNet1D(base=base, cond_dim=int(ck.get('cond_dim', 0)))
    model.load_state_dict(state)
    model.eval()
    diff = mod.Diffusion(T)

    d = np.load(a.data)
    Xraw = d['X']
    split = np.asarray(d['split']).astype(np.int8)
    gid = np.asarray(d['group_id'])
    S = torch.from_numpy(np.asarray(d['style']).astype(np.int64))
    occ = Xraw.reshape(len(Xraw), -1).mean(axis=1).astype(np.float32)
    cont = torch.from_numpy(np.clip((occ - dc['lo']) / max(1e-6, dc['hi'] - dc['lo']), 0, 1)
                           .astype(np.float32))
    idx = np.where(split == a.split)[0]
    if len(idx) == 0:
        raise SystemExit(f'[错误] split=={a.split} 没有窗口')
    _, inv = np.unique(gid[idx], return_inverse=True)
    lbl = torch.from_numpy(inv.astype(np.int64))
    Xe = torch.from_numpy(np.ascontiguousarray(Xraw[idx])).float()
    Se, De = S[idx], cont[idx]
    pooled, gm, vec, per_grp = eval_set(model, diff, pred, Xe, Se, De, lbl, T)
    names = {0: 'train', 1: 'val', 2: 'test'}
    print(f'\n split=={a.split}（{names[a.split]}）：{len(idx)} 窗 / '
          f'{len(np.unique(gid[idx]))} 谱 / {len(np.unique(inv))} 组')
    print(f'  池化损失 {pooled:.4f} / 组均 {gm:.4f}')
    sty = np.asarray(d['style'])[idx]
    per_style = {}
    for s in range(4):
        m = sty == s
        if m.sum():
            per_style[int(s)] = {'窗': int(m.sum()), '池化': round(float(vec[m].mean()), 6)}
            print(f'  style {s}: {int(m.sum())} 窗  池化 {float(vec[m].mean()):.4f}')
    res = {'ckpt': a.ckpt, 'split': a.split, 'split_name': names[a.split],
           'windows': int(len(idx)), 'charts': int(len(np.unique(gid[idx]))),
           'groups': int(len(np.unique(inv))),
           'pooled': round(pooled, 6), 'group_mean': round(gm, 6),
           'per_style': per_style, 'per_group': per_grp,
           'protocol': '固定 t/噪声 seed 1234；x0 目标未加权 MSE；同一 ckpt；'
                       '难度条件用 ckpt 的 lo/hi 复算'}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
