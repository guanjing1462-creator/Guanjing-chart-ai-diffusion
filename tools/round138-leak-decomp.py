#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round138 · A/B 泄漏分解：B 集里有多少窗口训练真的见过，见过的那部分损失低多少。

训练器每 epoch 用 draw_pool（组上限 + 均匀抽 n_target）从 split==0 抽 8000 窗，
种子 = seed*100003+ep；这里**按同样的规则、同样的种子**重建每轮的行号，取并集当
「训练见过的窗口」。然后把 B 集（seed+7919 抽出的 5025 窗）切成
  B_seen   = B ∩ seen
  B_unseen = B − seen
分别算损失，与 A（split==1，无泄漏）对比 —— 这条分解说明 B 的损失里有多少是
「同一批窗口被记住」而不是「跨谱泛化」。

用法：
  python3 tools/round138-leak-decomp.py --ckpt out/diffusion/round138-train/diffusion32-gj-grouped-last.pt \
      --out out/diffusion/round138-train/leak-decomp.json
"""
import argparse, importlib.util, json, os
import numpy as np
import torch

SEED = 0
GROUP_CAP = 300
N_EPOCH = 8000
AB_N = 5025


def draw_pool(grp_idx, n_target, seed, cap):
    """与 tools/train-diffusion32-grouped.py 的 draw_pool 逐行同构（同一 torch 随机序列）。"""
    g = torch.Generator().manual_seed(seed)
    parts = []
    for arr in grp_idx:
        m = len(arr)
        take = m if (not cap or m <= cap) else cap
        if take >= m:
            parts.append(torch.from_numpy(arr))
        else:
            sel = torch.randperm(m, generator=g)[:take]
            parts.append(torch.from_numpy(arr[sel.numpy()]))
    pool = torch.cat(parts)
    if len(pool) > n_target:
        pool = pool[torch.randperm(len(pool), generator=g)[:n_target]]
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='out/diff-data-grouped/dataset.npz')
    ap.add_argument('--trainer', default='tools/train-diffusion32-grouped.py')
    ap.add_argument('--epochs', type=int, default=9, help='实际训练轮数（重建 seen 用）')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    d = np.load(a.data)
    Xraw = d['X']
    split = np.asarray(d['split']).astype(np.int8)
    gid = np.asarray(d['group_id'])
    tr_np = np.where(split == 0)[0]
    va_np = np.where(split == 1)[0]
    _, inv_tr = np.unique(gid[tr_np], return_inverse=True)
    grp_idx = [tr_np[inv_tr == j] for j in range(len(np.unique(inv_tr)))]

    seen = np.zeros(len(Xraw), dtype=bool)
    sizes = []
    for ep in range(1, a.epochs + 1):
        pool = draw_pool(grp_idx, N_EPOCH, SEED * 100003 + ep, GROUP_CAP)
        seen[pool.numpy()] = True
        sizes.append(int(len(pool)))
    B = draw_pool(grp_idx, AB_N, SEED + 7919, GROUP_CAP).numpy()
    b_seen = B[seen[B]]
    b_unseen = B[~seen[B]]
    print(f'重建 seen：{a.epochs} 轮 × {N_EPOCH} 抽 → 去重后 {int(seen.sum())} 窗'
          f'（train {len(tr_np)} 窗的 {100 * seen.mean() / max(1e-9, (split == 0).mean()):.1f}%）')
    print(f'B 集 {len(B)} 窗：见过 {len(b_seen)}（{100 * len(b_seen) / len(B):.1f}%）'
          f' / 没见过 {len(b_unseen)}（{100 * len(b_unseen) / len(B):.1f}%）')
    print(f'B 覆盖 {len(np.unique(gid[B]))} 组，其中组内至少 1 窗被训过的组：'
          f'{len(np.unique(gid[b_seen])) if len(b_seen) else 0}')

    # ---- 损失：同一把尺子（seed 1234 的固定 t/噪声按各自长度生成，池化比较）----
    mods = importlib.util.spec_from_file_location('t32g', a.trainer)
    mod = importlib.util.module_from_spec(mods)
    mods.loader.exec_module(mod)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    dc = ck['difficulty_condition']
    model = mod.UNet1D(base=int(ck['base']), cond_dim=int(ck.get('cond_dim', 0)))
    model.load_state_dict(ck['state'])
    model.eval()
    T = int(ck.get('timesteps', 200))
    diff = mod.Diffusion(T)
    S = torch.from_numpy(np.asarray(d['style']).astype(np.int64))
    occ = Xraw.reshape(len(Xraw), -1).mean(axis=1).astype(np.float32)
    cont = torch.from_numpy(np.clip((occ - dc['lo']) / max(1e-6, dc['hi'] - dc['lo']), 0, 1)
                           .astype(np.float32))

    def loss_of(idx, tag):
        Xe = torch.from_numpy(np.ascontiguousarray(Xraw[idx])).float()
        n = len(idx)
        g = torch.Generator().manual_seed(1234)
        vt = torch.randint(0, T, (n,), generator=g)
        vn = torch.randn(n, 4, Xraw.shape[2], generator=g)
        vec = torch.zeros(n)
        with torch.no_grad():
            for i in range(0, n, 512):
                m = min(512, n - i)
                sl = slice(i, i + m)
                xt = diff.q(Xe[sl], vt[sl], vn[sl])
                p = model(xt, vt[sl], S[idx][sl], cont[idx][sl], None)
                _, tgt, _ = diff.target(Xe[sl], vt[sl], vn[sl], ck['pred'])
                vec[sl] = ((p - tgt) ** 2).mean(dim=(1, 2))
        v = float(vec.mean())
        print(f'  {tag:<28} {v:.4f}  （{n} 窗）', flush=True)
        return v

    print('\n损失（未加权 MSE，x0 目标）：')
    res = {'seen_windows': int(seen.sum()), 'train_windows': int(len(tr_np)),
           'seen_frac_of_train': round(float(seen[tr_np].mean()), 4),
           'B_total': int(len(B)), 'B_seen': int(len(b_seen)), 'B_unseen': int(len(b_unseen)),
           'A': {}, 'B': {}, 'B_seen': {}, 'B_unseen': {}, 'train_pool_last': {}}
    a_loss = loss_of(va_np, 'A  split==1（无泄漏 val）')
    res['A']['pooled'] = round(a_loss, 6)
    res['B']['pooled'] = round(loss_of(B, 'B  split==0 抽样（泄漏式）'), 6)
    if len(b_seen):
        res['B_seen']['pooled'] = round(loss_of(b_seen, 'B_seen（训练见过）'), 6)
    if len(b_unseen):
        res['B_unseen']['pooled'] = round(loss_of(b_unseen, 'B_unseen（训练没见过）'), 6)
    last_pool = draw_pool(grp_idx, N_EPOCH, SEED * 100003 + a.epochs, GROUP_CAP).numpy()
    res['train_pool_last']['pooled'] = round(loss_of(last_pool, '末轮训练池（8000 窗）'), 6)
    if len(b_seen) and len(b_unseen):
        res['seen_minus_unseen'] = round(res['B_seen']['pooled'] - res['B_unseen']['pooled'], 6)
        print(f'\n  B_seen − B_unseen = {res["seen_minus_unseen"]:+.4f}（负值=见过更准，即记忆红利）')
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f'→ {a.out}')


if __name__ == '__main__':
    main()
