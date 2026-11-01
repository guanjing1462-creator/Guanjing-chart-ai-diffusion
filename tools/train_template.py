#!/usr/bin/env python3
"""
train_template.py — 谱面语言模型的训练模板（未在本项目跑过）

定位：这是「阶段 3」的脚手架，不是能直接出好结果的成品。
     跑之前请先确认 06-chart-ai/README.md 里的数据量门槛。

依赖：
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    # 单卡 4090 约 24G 显存，20M 参数模型 batch 64 够用

数据：
    node tools/build-dataset.mjs <谱面目录> -o out/dataset.jsonl

用法：
    python tools/train_template.py --data out/dataset.jsonl --steps 20000
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 数据
def load_jsonl(path, keys_filter=4, max_frames=8192):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["keys"] != keys_filter:
                continue
            toks = d["tokens"][:max_frames]
            if len(toks) < 64:
                continue
            samples.append(toks)
    return samples


def make_batch(samples, batch_size, seq_len, vocab, device):
    """随机裁一段连续窗口，避免整首塞进显存。"""
    xs, ys = [], []
    for _ in range(batch_size):
        s = random.choice(samples)
        start = random.randint(0, max(0, len(s) - seq_len - 1))
        chunk = s[start:start + seq_len + 1]
        if len(chunk) < seq_len + 1:
            chunk = chunk + [0] * (seq_len + 1 - len(chunk))
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


# ---------------------------------------------------------------- 模型
class ChartGPT(nn.Module):
    """decoder-only Transformer。

    关键设计：
      - 词表很小（4K 只有 16），所以 embedding 和输出头都很轻
      - 条件输入（目标密度 / 段位）用 prefix token 注入，见 cond_dim
      - 这里只给了无条件版本；加条件时把 cond 向量投影后拼到 position embedding 上
    """

    def __init__(self, vocab, d_model=384, n_layer=6, n_head=6, max_len=8192, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

        self.max_len = max_len
        self.register_buffer(
            "causal", torch.triu(torch.ones(max_len, max_len), diagonal=1).bool(), persistent=False
        )

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.tok(x) + self.pos(pos))
        h = self.blocks(h, mask=self.causal[:T, :T])
        return self.head(self.ln(h))


# ---------------------------------------------------------------- 采样
@torch.no_grad()
def sample(model, prompt, n_new, temperature=0.95, top_k=8, device="cuda"):
    """temperature/top_k 都要调：谱面 token 分布很尖，不限制会一直吐同一个音型。"""
    model.eval()
    seq = torch.tensor([prompt], dtype=torch.long, device=device)
    for _ in range(n_new):
        ctx = seq[:, -model.max_len:]
        logits = model(ctx)[:, -1, :] / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, nxt], dim=1)
    model.train()
    return seq[0].tolist()


# ---------------------------------------------------------------- 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/dataset.jsonl")
    ap.add_argument("--out", default="out/chartgpt.pt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--keys", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_jsonl(args.data, keys_filter=args.keys)
    if not samples:
        raise SystemExit(f"{args.data} 里没有 {args.keys}K 样本，先跑 build-dataset.mjs")

    vocab = 1 << args.keys
    total_frames = sum(len(s) for s in samples)
    print(f"样本 {len(samples)} 个 / {total_frames:,} 帧 / 词表 {vocab}")

    # 数据量门槛检查：参数量的 100 倍 token 才比较稳
    est_params = 384 * 384 * 6 * 12 // 1000  # 粗略
    if total_frames < est_params * 100:
        print(f"⚠ 当前 {total_frames:,} 帧，建议至少 {est_params * 100:,} 帧才训 20M 参数模型")
        print("  继续训练会严重过拟合，建议先扩数据（抓 ranked 谱面）")

    model = ChartGPT(vocab).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_param/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = make_batch(samples, args.batch, args.seq, vocab, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0:
            ppl = math.exp(min(loss.item(), 20))
            print(f"step {step:6d}  loss {loss.item():.4f}  ppl {ppl:.2f}  lr {sched.get_last_lr()[0]:.2e}")
        if step % 5000 == 0:
            torch.save({"model": model.state_dict(), "vocab": vocab, "keys": args.keys}, args.out)
            print(f"  已保存 {args.out}")

    torch.save({"model": model.state_dict(), "vocab": vocab, "keys": args.keys}, args.out)

    # 生成一段样例，直接写成 dataset 格式，可以喂给 preview.html 看
    out_seq = sample(model, [0] * 8, 1200, device=device)
    with open("out/generated.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": "gen", "keys": args.keys, "bpm": 180, "frames": len(out_seq),
            "stepMs": 83.333, "tokens": out_seq, "holds": [],
            "meta": {"title": "model sample", "notes": sum(bin(t).count("1") for t in out_seq)},
        }) + "\n")
    print("样例已写到 out/generated.jsonl")


if __name__ == "__main__":
    main()
