#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 145 轮汇总：把 4 份 eval-<style>.json + checkpoint 元数据拼成报告数据。

只读 + 统计，不训练、不采样。输出 report-data.json 并打印 markdown 表格。
"""
import argparse, json, os
import torch

STYLES = ['tech', 'jack', 'speed', 'stamina']
DATASET_P132 = [0.5103, 0.0608, 0.6139, 0.1160]
DATASET_OCC = [0.1924, 0.2259, 0.1939, 0.1780]
BASE_P132 = [0.3962, 0.0923, 0.3562, 0.0208]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='out/diffusion/round145-perstyle')
    ap.add_argument('--baseline-ckpt', default='out/diffusion/round138-train/diffusion32-gj-grouped.pt')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    base = torch.load(a.baseline_ckpt, map_location='cpu', weights_only=False)
    base_params = sum(v.numel() for v in base['state'].values() if v.is_floating_point())
    out = {'基线单模型': {'ckpt': a.baseline_ckpt, 'params': base_params,
                        'val': base.get('val'), 'epoch': base.get('epoch'),
                        'batch': base.get('batch'), 'timesteps': base.get('timesteps'),
                        'train_windows': base.get('train_windows'),
                        'train_seconds': base.get('train_seconds'),
                        'p132': dict(zip(STYLES, BASE_P132))},
           '数据集p132': dict(zip(STYLES, DATASET_P132)),
           '数据集占位率': dict(zip(STYLES, DATASET_OCC)),
           '模型': {}}
    tot_params = tot_sec = 0
    for i, s in enumerate(STYLES):
        fp = os.path.join(a.root, f'{s}.pt')
        ej = os.path.join(a.root, f'eval-{s}.json')
        if not (os.path.exists(fp) and os.path.exists(ej)):
            print(f'[缺失] {s}: ckpt={os.path.exists(fp)} eval={os.path.exists(ej)}')
            continue
        ck = torch.load(fp, map_location='cpu', weights_only=False)
        ev = json.load(open(ej, encoding='utf-8'))['models'][s]
        p = ev['params']
        tot_params += p
        tot_sec += float(ck.get('train_seconds') or 0)
        row = {'params': p, 'val': ck.get('val'), 'epoch': ck.get('epoch'),
               'batch': ck.get('batch'), 'timesteps': ck.get('timesteps'),
               'train_seconds': ck.get('train_seconds'),
               'train_windows': ck.get('train_windows'),
               'val_windows': ck.get('val_windows'),
               'train_groups': ck.get('train_groups'),
               'difficulty_lo_hi': [ck['difficulty_condition']['lo'],
                                    ck['difficulty_condition']['hi']],
               'own': ev['own'], 'common': ev.get('common'),
               'dataset_p132': DATASET_P132[i], 'dataset_occ': DATASET_OCC[i],
               'baseline_p132': BASE_P132[i],
               'delta': round(ev['own']['p132'] - DATASET_P132[i], 4),
               'ok': abs(ev['own']['p132'] - DATASET_P132[i]) <= 0.05}
        out['模型'][s] = row
    out['总参数量_4模型'] = tot_params
    out['总训练秒'] = round(tot_sec, 1)
    out['总训练分钟'] = round(tot_sec / 60, 1)
    out['达标数'] = sum(1 for r in out['模型'].values() if r['ok'])

    print('=== 模型配置与训练 ===')
    print('| 风格 | 参数量 | val（池化） | 训练窗/轮 | epoch | 训练时长 | 难度条件 lo–hi |')
    print('|---|---:|---:|---:|---:|---:|---|')
    for s in STYLES:
        r = out['模型'].get(s)
        if not r:
            continue
        print(f'| {s} | {r["params"]/1e6:.4f}M | {r["val"]:.4f} | {r["train_windows"]} | '
              f'{r["epoch"]} | {r["train_seconds"]:.0f}s（{r["train_seconds"]/60:.1f}min） | '
              f'{r["difficulty_lo_hi"][0]:.3f}–{r["difficulty_lo_hi"][1]:.3f} |')
    print(f'| 合计 | {tot_params/1e6:.4f}M | — | — | — | {tot_sec:.0f}s（{tot_sec/60:.1f}min） | — |')
    print('\n=== 四风格 p132 三方对照 ===')
    print('| 风格 | 数据集 p132 | 单模型 baseline | 分模型 | Δ（分模型−数据集） | 达标 |')
    print('|---|---:|---:|---:|---:|---|')
    for i, s in enumerate(STYLES):
        r = out['模型'].get(s)
        if not r:
            print(f'| {s} | {DATASET_P132[i]:.4f} | {BASE_P132[i]:.4f} | 未训练 | — | — |')
            continue
        print(f'| {s} | {DATASET_P132[i]:.4f} | {BASE_P132[i]:.4f} | **{r["own"]["p132"]:.4f}** | '
              f'{r["delta"]:+.4f} | {"✅" if r["ok"] else "❌"} |')
    print(f'\n达标 {out["达标数"]}/{len(out["模型"])}；'
          f'4 模型总参数 {tot_params/1e6:.4f}M（单模型 {base_params/1e6:.4f}M，'
          f'倍数 {tot_params/base_params:.2f}×）；总训练 {tot_sec/60:.1f} 分钟')

    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f'→ {a.out}')


if __name__ == '__main__':
    main()
