#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩散模型的「占位率 / 结构」评估器（阈值细扫 + 确定性尾巴 + 多 seed）。

为什么要有它
------------
`tools/sample-diffusion.py` 每次只能用一个阈值、且必须重新跑 200 步去噪；
一次采样已经有 200 步 × N 窗口的成本，细扫阈值不该重采。本脚本：
  1. 采样一次 → 拿到连续 x0 概率卷 P（[n,4,128]）；
  2. 在同一份 P 上扫一串阈值（0.2~0.6，步长 0.01），给出「占位率 / 每行音符
     数 / 空行比例 / 行分布 / 同列相邻行相关」随阈值的变化；
  3. 给出「命中目标占位率」的校准阈值（默认 0.407 = 训练集全局占位率）；
  4. `--det-tails` 可对比随机祖先采样与确定性尾巴（t<K 不加噪）；
  5. `--seeds` 多个种子，避免用单个 2 窗口的运气当结论。

权重从 `--ckpt` 读；模型结构复用 `tools/train-density.py`（与官方采样器
逐字节等价的 UNet1D），所以评估结果和官方采样器一致。
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        'train_density', os.path.join(HERE, 'train-density.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


td = _load_train_module()


def load_model(ckpt, threads=1, state_key='state'):
    torch.set_num_threads(max(1, threads))
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    if state_key != 'state' and isinstance(ck, dict) and state_key in ck:
        state = ck[state_key]
    else:
        state = ck['state'] if (isinstance(ck, dict) and 'state' in ck) else ck
    base = int(ck.get('base', state['inp.weight'].shape[0])) if isinstance(ck, dict) else 48
    temb = 128
    if 'temb.0.weight' in state:
        temb = int(state['temb.0.weight'].shape[1])
    pred = (ck.get('pred', 'eps') if isinstance(ck, dict) else 'eps')
    timesteps = int(ck.get('timesteps', 200)) if isinstance(ck, dict) else 200
    model = td.UNet1D(cin=4, base=base, temb_dim=temb)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, td.Diffusion(timesteps), {'pred': pred, 'base': base,
                                            'timesteps': timesteps,
                                            'ckpt_val': ck.get('val') if isinstance(ck, dict) else None,
                                            'epoch': ck.get('epoch') if isinstance(ck, dict) else None,
                                            'density_at_save': ck.get('density_at_save') if isinstance(ck, dict) else None}


def struct_stats(P, th):
    B = (P > th)
    rs = B.sum(1)                                    # [n,128]
    h = np.bincount(rs.ravel(), minlength=5) / rs.size
    corr = float((B[:, :, 1:] * B[:, :, :-1]).mean())
    return {
        '占位率': round(float(B.mean()), 4),
        '每行音符': round(float(rs.mean()), 3),
        '空行比例': round(float((rs == 0).mean()), 3),
        '行分布0-4': [round(float(v), 3) for v in h],
        '同列相邻行': round(corr, 4),
        '各列占位率': [round(float(v), 4) for v in B.mean(axis=(0, 2))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='out/models/diffusion4k-x0full.pt')
    ap.add_argument('--style', type=int, default=1)
    ap.add_argument('--diff', type=int, default=2)
    ap.add_argument('--num', type=int, default=4, help='每个 seed 采几个窗口')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--det-tails', default='0', help='逗号分隔的 K：t<K 用确定性均值')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--target', type=float, default=0.4072, help='目标占位率（校准阈值用）')
    ap.add_argument('--outdir', default='out/diffusion/density')
    ap.add_argument('--tag', default='')
    ap.add_argument('--save-p', action='store_true', help='保存连续概率卷 npz（供离线细扫）')
    ap.add_argument('--state', default='state', choices=['state', 'state_raw'],
                    help='读哪份权重：state=EMA（采样器默认读的），state_raw=原始权重')
    a = ap.parse_args()

    ths = np.round(np.arange(0.20, 0.601, 0.01), 2)
    seeds = [int(x) for x in a.seeds.split(',') if x.strip() != '']
    det_tails = [int(x) for x in a.det_tails.split(',') if x.strip() != '']

    model, diff, info = load_model(a.ckpt, a.threads, a.state)
    print(f'权重 {a.ckpt}（{a.state}）  pred={info["pred"]} base={info["base"]} T={info["timesteps"]} '
          f'epoch={info["epoch"]} ckpt_val={info["ckpt_val"]}', flush=True)
    print(f'条件 style={a.style} diff={a.diff}｜num={a.num} seeds={seeds} det_tails={det_tails}', flush=True)

    out = {'ckpt': a.ckpt, 'state_key': a.state, 'info': info, 'style': a.style, 'diff': a.diff,
           'num': a.num, 'seeds': seeds, 'det_tails': det_tails,
           'target': a.target, 'runs': []}
    os.makedirs(a.outdir, exist_ok=True)

    for dt in det_tails:
        Ps = []
        t0 = time.time()
        for sd in seeds:
            torch.manual_seed(sd)
            P = td.ddpm_sample(model, diff, a.num, a.style, a.diff,
                               pred=info['pred'], det_tail=dt).numpy()
            Ps.append(P)
        P = np.concatenate(Ps, 0).astype(np.float32)
        cost = time.time() - t0
        tag = f's{a.style}d{a.diff}-dt{dt}-n{len(seeds) * a.num}'
        run = {'det_tail': dt, 'windows': int(P.shape[0]), 'sec': round(cost, 1),
               'x0均值': round(float(P.mean()), 4), 'x0标准差': round(float(P.std()), 4),
               'x0q10': round(float(np.quantile(P, .10)), 4),
               'x0q50': round(float(np.quantile(P, .50)), 4),
               'x0q90': round(float(np.quantile(P, .90)), 4),
               'sweep': {}}
        for th in ths:
            run['sweep'][f'{th:.2f}'] = struct_stats(P, float(th))
        # 校准阈值：占位率最接近 target 的那个阈值
        occs = {float(th): run['sweep'][f'{th:.2f}']['占位率'] for th in ths}
        best_th = min(occs, key=lambda k: abs(occs[k] - a.target))
        run['校准'] = {'阈值': best_th, '占位率': occs[best_th],
                       '每行音符': run['sweep'][f'{best_th:.2f}']['每行音符'],
                       '结构': run['sweep'][f'{best_th:.2f}']}
        run['阈值0.50'] = run['sweep']['0.50']
        out['runs'].append(run)

        print(f'\n=== det_tail={dt}（{P.shape[0]} 窗口，{cost:.0f}s）x0 均值 {P.mean():.4f} '
              f'std {P.std():.4f} q50 {np.quantile(P, .5):.4f}', flush=True)
        print(f'{"阈值":>5} {"占位率":>7} {"每行音符":>8} {"空行":>6} {"行分布0-4":>22} {"同列相邻":>8}', flush=True)
        for th in ths:
            s = run['sweep'][f'{th:.2f}']
            mark = ' ← 校准' if abs(float(th) - best_th) < 1e-9 else ''
            print(f'{th:5.2f} {s["占位率"]:7.4f} {s["每行音符"]:8.3f} {s["空行比例"]:6.3f} '
                  f'{str(s["行分布0-4"]):>22} {s["同列相邻行"]:8.4f}{mark}', flush=True)
        print(f'校准阈值 {best_th:.2f} → 占位率 {occs[best_th]:.4f}（目标 {a.target}）', flush=True)

        if a.save_p:
            p = os.path.join(a.outdir, f'P-{a.tag or tag}.npz')
            np.savez_compressed(p, P=P, style=np.int64(a.style), diff=np.int64(a.diff),
                                det_tail=np.int64(dt), seeds=np.array(seeds),
                                ckpt=a.ckpt)
            print(f'  概率卷 → {p}', flush=True)

    tag = a.tag or f'{os.path.basename(a.ckpt).replace(".pt", "")}-s{a.style}d{a.diff}'
    jp = os.path.join(a.outdir, f'eval-{tag}.json')
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\n评估结果 → {jp}', flush=True)


if __name__ == '__main__':
    main()
