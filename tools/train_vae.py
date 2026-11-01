#!/usr/bin/env python3
"""KL-VAE：谱面张量 [16,4096] → [16,512]（时间 8× 压缩），Mug-Diffusion first stage 同款。

    python3 tools/train_vae.py --data out/mug_tensors --out out/vae --epochs 8

CPU 可跑（一晚量级）。结构对齐 configs/mug/mug_diffusion.yaml 的 ddconfig：
channel_mult [1,2,4,4]（3 次下采样），num_res_blocks 1，z_channels 16。
重建损失分通道加权（Mug 注释掉的 ManiaReconstructLoss 参数）：
  start 通道 L1、holding 通道 ×0.5、offset 通道 ×0.5、尾偏移 ×0.2。
验收指标：解码二值化后 start 通道的 F1（>0.99 才算"潜空间可逆"）。
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resblock(ch_in, ch_out):
    return nn.Sequential(
        nn.GroupNorm(8, ch_in), nn.SiLU(),
        nn.Conv1d(ch_in, ch_out, 3, padding=1),
        nn.GroupNorm(8, ch_out), nn.SiLU(),
        nn.Conv1d(ch_out, ch_out, 3, padding=1),
    )


class Encoder(nn.Module):
    def __init__(self, x_ch=16, mid=64, z=16, mult=(1, 2, 4, 4)):
        super().__init__()
        self.conv_in = nn.Conv1d(x_ch, mid, 3, padding=1)
        chs = [mid * m for m in mult]
        self.down = nn.ModuleList()
        for i in range(len(mult) - 1):
            ch_in = chs[i - 1] if i > 0 else chs[0]   # 输入来自上一级下采样输出
            self.down.append(nn.ModuleDict({
                'block': resblock(ch_in, chs[i]),
                'down': nn.Conv1d(chs[i], chs[i], 4, stride=2, padding=1),
            }))
        self.mid = resblock(chs[-1], chs[-1])
        self.to_mu = nn.Conv1d(chs[-1], z, 1)
        self.to_logvar = nn.Conv1d(chs[-1], z, 1)

    def forward(self, x):
        h = self.conv_in(x)
        for d in self.down:
            h = d['down'](d['block'](h))
        h = self.mid(h)
        return self.to_mu(h), self.to_logvar(h).clamp(-15, 15)


class Decoder(nn.Module):
    def __init__(self, x_ch=16, mid=64, z=16, mult=(1, 2, 4, 4)):
        super().__init__()
        chs = [mid * m for m in mult]
        self.conv_in = nn.Conv1d(z, chs[-1], 1)
        self.up = nn.ModuleList()
        for i in range(len(mult) - 1, 0, -1):
            self.up.append(nn.ModuleDict({
                'up': nn.Upsample(scale_factor=2, mode='nearest'),
                'block': resblock(chs[i], chs[i - 1]),
            }))
        self.mid = resblock(chs[0], chs[0])
        self.conv_out = nn.Conv1d(chs[0], x_ch, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        for u in self.up:
            h = u['block'](u['up'](h))
        h = self.mid(h)
        return self.conv_out(h)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def reparam(self, mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    # 通道权重：start(0-4) 1.0 / start_off(4-8) 0.5 / holding(8-12) 0.5 / end_off(12-16) 0.2
    W = torch.tensor([1.0] * 4 + [0.5] * 4 + [0.5] * 4 + [0.2] * 4).view(16, 1)


# 通道角色：0-4 is_start（二值）、4-8 头偏移（L1）、8-12 LN 体（二值）、12-16 尾偏移（L1）
BIN_CH = [0, 1, 2, 3, 8, 9, 10, 11]
OFF_CH = [4, 5, 6, 7, 12, 13, 14, 15]
POS_WEIGHT = 64.0   # 正类占比 ~1-2%，pos_weight=(1-p)/p ≈ 64


def vae_loss(x, recon, mu, logvar, kl_weight=1e-6):
    """v3：二值通道用带正类权重的 BCE-with-logits，偏移通道用 L1（仅在有音符处计）。

    旧版全通道 L1 的病根：正类占比 ~1.5%，L1 均值被 98% 的零目标主导，
    模型收敛到「全输出 0」——loss 看着很低（0.009 ≈ 音符密度），F1 永远
    爬不动。BCE+pos_weight 直接优化「把稀疏的 1 找回来」。"""
    rec_bin = 0.0
    for ch in BIN_CH:
        col = ch % 4
        is_start = (ch < 4)
        if is_start:
            mask = torch.ones_like(x[:, ch])
        else:
            # LN 体通道只在头出现过的地方有意义
            mask = (x[:, ch - 4] + x[:, ch]).clamp(0, 1)
        if mask.sum() < 1:
            continue
        rec_bin = rec_bin + F.binary_cross_entropy_with_logits(
            recon[:, ch], x[:, ch],
            pos_weight=torch.tensor(POS_WEIGHT, device=x.device),
            reduction='mean')
    rec_off = 0.0
    for ch in [4, 5, 6, 7, 12, 13, 14, 15]:
        base = ch - 4 if ch < 8 else ch - 4
        mask = x[:, base] if ch < 8 else x[:, base]  # 头偏移挂在 is_start 上
        m = (x[:, base] > 0.5).float() if ch < 8 else x[:, base - 4]
        m = (x[:, base] > 0.5).float()
        if m.sum() < 1:
            continue
        rec_off = rec_off + ((recon[:, ch] - x[:, ch]).abs() * m).sum() / m.sum()
    rec = (rec_bin + rec_off) / 12.0
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + kl_weight * kl, rec, kl


def load_batch(data_dir, idx_list, batch=8, seed=None):
    rng = np.random.RandomState(seed)
    files = [idx_list[i] for i in rng.permutation(len(idx_list))[:batch]]
    xs = []
    for f in files:
        d = np.load(f)
        xs.append(d['note'].astype(np.float32))
    return torch.from_numpy(np.stack(xs))


def evaluate(model, files, n=40):
    """解码二值化后的 start 通道 F1 + LN 体 F1。"""
    model.eval()
    f1s, f1h = [], []
    with torch.no_grad():
        for f in files[:n]:
            x = torch.from_numpy(np.load(f)['note'].astype(np.float32)).unsqueeze(0)
            mu, logvar = model.enc(x)
            recon = model.dec(mu)
            for ch_off, out in [(0, f1s), (8, f1h)]:
                pred = (torch.sigmoid(recon[0, ch_off:ch_off+4]) > 0.5).float()
                true = (x[0, ch_off:ch_off+4] > 0.5).float()
                tp = (pred * true).sum()
                f1s_out = 2 * tp / (pred.sum() + true.sum() + 1e-6)
                out.append(float(f1s_out))
    model.train()
    return float(np.mean(f1s)), float(np.mean(f1h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='out/mug_tensors')
    ap.add_argument('--out', default='out/vae')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--kl', type=float, default=1e-6)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--cap-steps', type=int, default=0,
                    help='每 epoch 最多多少步（0=全量）。沙箱活跃窗口 ~10 分钟，'
                         'CPU 一步 ~0.35s，设 1400 正好一个窗口。')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    if not files:
        sys.exit(f'{args.data} 下没有张量，先跑 build-mug-tensors.py')
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, 'vae.pt')
    model = VAE()
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        print('续训：载入', ckpt)
    # 阶梯衰减：每 4 epoch lr ×0.7——0.95 平台后的振荡（epoch4 跌到
    # 0.85）是 lr 偏高的典型症状，细水长流比反复冲高更能逼近 0.99
    def lr_at(ep):
        return args.lr * (0.7 ** (ep // 4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr_at(0))

    f1s, f1h = evaluate(model, files)
    print(f'初始: start F1 {f1s:.4f}  holding F1 {f1h:.4f}  ({len(files)} 张张量可用)')
    if args.eval_only:
        return

    step = 0
    per_epoch = max(1, len(files) // args.batch)
    for ep in range(args.epochs):
        tot = 0.0
        n_steps = per_epoch if not args.cap_steps else min(per_epoch, args.cap_steps)
        import time as _t
        for g in opt.param_groups:
            g['lr'] = lr_at(ep)
        for it in range(n_steps):
            _ts = _t.time()
            x = load_batch(args.data, files, args.batch, seed=step)
            _t_load = _t.time() - _ts
            _ts = _t.time()
            mu, logvar = model.enc(x)
            z = model.reparam(mu, logvar)
            recon = model.dec(z)
            loss, rec, kl = vae_loss(x, recon, mu, logvar, args.kl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            if it % 20 == 0:
                print(f'  it={it} 取数 {_t_load:.2f}s 步 {(_t.time()-_ts):.2f}s', flush=True)
            # 每 20 步存档 + 打 F1：只在 epoch 末存档 = 窗口一断进度全丢。
            # （历史 bug：step 每轮 +2 导致 step%20 永远为奇数命中不到，
            #   实测 1.5 小时没有任何存档/F1 日志，已修）
            if step % 20 == 0:
                f1s, f1h = evaluate(model, files, n=24)
                print(f'step {step}: loss {tot/(it+1):.4f}  start F1 {f1s:.4f}  holding F1 {f1h:.4f}', flush=True)
                torch.save(model.state_dict(), ckpt)
                tot = 0.0
            step += 1
        f1s, f1h = evaluate(model, files)
        print(f'epoch {ep} ({n_steps} 步): loss {tot/per_epoch:.4f}  start F1 {f1s:.4f}  holding F1 {f1h:.4f}')
        torch.save(model.state_dict(), ckpt)
    print('最终检查点:', ckpt)


if __name__ == '__main__':
    main()
