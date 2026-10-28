# YAMERO 谱面生成 AI —— 全量交接仓库

4K 谱面生成 AI 全套资产：**音频条件扩散（Mug-Diffusion 自研等价物）+ KL-VAE + 36837 张谱面数据集 + 规则管线（音频骨架→风格指纹着色→站点闭环锚定）**。

## 必读顺序

1. **[HANDOFF.md](HANDOFF.md)** —— 全量交接文档（数据格式/训练状态/出谱命令/12 条踩坑/站点刻度经验）
2. **[NEXT-STEPS.md](NEXT-STEPS.md)** —— 后续任务清单（P0/P1/P2，含验收标准）
3. [RUNBOOK.md](RUNBOOK.md) / [TRAINING-PLAN.md](TRAINING-PLAN.md) / [FEATURES.md](FEATURES.md) / [MUG-DIFFUSION-剖析.md](MUG-DIFFUSION-剖析.md)

## 仓库内容（代码与文档）

```
src/  tools/            # 规则管线与训练全部代码
HANDOFF.md 等 6 份文档   # 交接与手册
generator.html preview.html notes.json   # chartgen 分析器（骨架的 analysis.json 来源）
libminacalc.so          # minacalc 难度库（anchor-difficulty.py 依赖）
portal.html             # 自包含交接中心网页（浏览器直接打开）
```

## 模型权重与数据集 → 见 [Releases v1.0-assets](../../releases/tag/v1.0-assets)

| 资产 | 体积 | 内容 |
|---|---|---|
| 01-code-docs-models.tar.gz | 79M | 代码快照 + VAE(vae.pt F1 0.975) + 扩散(diff.pt loss 0.108) 权重 + 36 桶指纹 + 日志 |
| 02-site-analyzer.tar.gz | 27M | 站点 Koishi 分析器（D 值/pattern 校验） |
| 03-dataset-charts.tar.gz | 1.2G | 36837 张谱面 + 36598 张 21 维控制特征 |
| d-tensors.tar.gz | 313M | 36837 张 Mug 张量 [16,4096] |
| d-audio-1/2.tar.gz | 2.4G | 9498 张音频 mel [128,4096] |
| p-pool-1/2/3.tar.gz | 4.2G | 原始音频池（音频对训练源） |
| 06-final-charts.tar.gz | 60M | v1→v10 全部成品 osz |

每个 .tar.gz **独立可解压**，全部解到同一目录即恢复完整项目（解压结构见 HANDOFF.md §1）。

## 快速开始

```sh
tar xzf 01-code-docs-models.tar.gz          # 代码+文档+权重
python3 tools/skeleton-from-audio.py --analysis out/final14-s50/analysis.json -o /tmp/sk.osu
python3 tools/pattern-painter.py --skeleton /tmp/sk.osu --style rice/stream/high --dps 10.3 --version test -o out.osu
python3 tools/anchor-difficulty.py --skeleton /tmp/sk.osu --style rice/technical/high --target 15.8 --version t -o final.osu
```
