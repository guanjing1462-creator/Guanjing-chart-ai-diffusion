# 后续任务清单（NEXT-STEPS）

> 交接时刻：2026-10-29 02:42。与 HANDOFF.md §8 状态快照衔接，按优先级排序。
> 每个任务给出：目标 / 具体步骤 / 验收标准 / 预估工作量。

---

## P0-A：音频特征提取收尾（沙箱正在跑，接管后先看一眼）

- **现状**：9498 个 mel / 预计上限 ~1 万（36837 张谱面里约 25% 有音频，其余是 no_audio 跳过）
- **步骤**：`cd 06-chart-ai && nohup python3 -u tools/build-audio-features.py --workers 2 > out/audio-extract.log 2>&1 &`（断点续传，自动跳过已完成）
- **验收**：`ls out/audio_features | wc -l` 停止增长且 _index.jsonl 行数一致；`待处理`归零
- **注意**：死锁就降 `--workers 1`（已内置音频索引，每张 ~2s）
- 工作量：剩余 <500 张，~20 分钟

## P0-B：扩散训练推进 + 踩音率对比（本项目的核心里程碑）

- **现状**：epoch0 loss 0.1083（9166 音频对）；规则管线骨架踩音率 100% vs 扩散骨架 70.4%（历史基线）
- **步骤**：
  1. 续训：`nohup python3 -u tools/train_diffusion.py > out/diffusion-train.log 2>&1 &`（自动载入 out/diffusion/diff.pt）
  2. 每 2-3 个 epoch 用当前 ckpt 采样一次：`python3 tools/train_diffusion.py --sample --ckpt out/diffusion/diff.pt --mel <INTERNET YAMERO 的 mel npz> ...`（脚本自带 --sample 模式，DDIM 50 步 + CFG 3）
  3. 采样产物做**踩音率对比实验**：与音频 onset 对齐（用 site-analysis 里的 onset 工具或 analysis.json 的 peaks），统计"音符落在 onset ±41ms 内比例"
- **验收**：loss 稳定 <0.10 且采样踩音率 **≥85%** → 扩散骨架 + 指纹着色混合管线接管交付线；<85% 则继续喂音频对（收 P0-A 的尾）再训
- 工作量：GPU 半天；CPU 上 1-2 天

## P1-A：VAE 爬到验收线

- **现状**：start F1 0.975 / holding F1 0.921（holding recall 100%，实际可用）
- **步骤**：`nohup python3 -u tools/train_vae.py > out/vae-train.log 2>&1 &`（自动续训；每 epoch 末打印 F1）
- **验收**：start F1 ≥0.99（TRAINING-PLAN 的验收线）；0.98+ 时可先放行扩散训练用当前冻结权重
- 工作量：每个 epoch ~30-60 分钟（4 核），预计 3-5 epoch

## P1-B：规则管线补全键型覆盖（用户点名"我要什么键型就要什么"）

- **现状**：v10 验证了 stream/technical/ln-technical 三条；**chordjack / jackspeed / handstream 桶未出过终版**
- **步骤**：
  1. `--style rice/chordjack/high --target 15.5` 走 §5 标准链路（锚定器会自动迭代）
  2. 站点验收看 pattern 是否判 **jack**（历史上 v12 的病是 jack 98.9%，合法目标应该是 chordjack 维度占优但 <30% 纯 jack）；同列间隔分布用质检脚本（HANDOFF §5 有 4K 列质检代码样例）
  3. jackspeed/handstream 同理；LN 系注意 ln_ratio token 与站点 RC_LN_RATIO_LIMIT=0.18
- **验收**：站点 pattern 判定与所选桶一致 + jack(<70ms) 比例人类范围（<20%）+ 无 LN 墙
- 工作量：每键型 1-2 小时（含站点迭代）

## P1-C：难度桶间插值（任意 D 精确锚定）

- **现状**：割线法已能锚到 ±0.15，但每次要跑站点迭代（每轮 ~50s）
- **步骤**：给 anchor-difficulty.py 加 `--cal-cache`：每个 style 桶首次锚定后把 (dps, D) 点写进 CAL 表持久化（out/cal-<style>.json），第二次同桶锚定直接插值初值，1-2 轮收敛
- **验收**：同桶第二次锚定 ≤2 轮收敛
- 工作量：~1 小时

## P2-A：换歌泛化（管线成熟后的必答题）

- **现状**：全链路只对 INTERNET YAMERO 验证过；骨架依赖 chartgen 分析器产物
- **步骤（换歌流程）**：
  1. **音频分析**：打开项目根的 `generator.html`（浏览器），加载新歌 ogg/mp3 → 产出该歌的 `analysis.json`（含 bpm/offsetMs/stepMs/peaks/accent 网格能量）——这就是 skeleton-from-audio.py 的输入
  2. `--analysis <新歌 analysis.json>` 走 §5 标准链路
  3. **指纹库无需换**（36 桶来自 3.6 万张谱面，风格是全局的）；但 CAL 锚定表要重扫（换歌 D 响应曲线不同，HANDOFF §4 的斜率经验是 185BPM 特化的）
- **验收**：新歌踩音 ≥95%、站点 D 锚定收敛、听感键型正确
- 工作量：半天（主要是人工听感验收）

## P2-B：正式 Mug 架构大训练（有 GPU 时）

- **现状**：自研等价物在 4 核 CPU 上训练；Mug-Diffusion 原架构（Mug Version 2/3 数据管线 + 3D UNet + clap token）未上 GPU
- **步骤**：GPU 环境下按 RUNBOOK 阶段 1-3 全量重跑：VAE batch 16 全量步 → 扩散 batch 16 全量；rate 增强 0.9-1.1；21 token 控制保持
- **验收**：D 值随 ett token 单调、pattern 判定随 pattern token 变化（RUNBOOK 阶段 3 的验收标准）
- 工作量：4090 级半天-一天

## 常备运维命令（接管沙箱先跑）

```sh
cd /workspace/dsh000/06-chart-ai
# 三线全开（断点续训）
nohup python3 -u tools/train_vae.py > out/vae-train.log 2>&1 &
nohup python3 -u tools/train_diffusion.py > out/diffusion-train.log 2>&1 &
nohup python3 -u tools/build-audio-features.py --workers 2 > out/audio-extract.log 2>&1 &
# 看进度
tail -3 out/vae-train.log out/diffusion-train.log out/audio-extract.log
# 杀进程防自匹配自杀：pkill -f "train[.]py" 这种点号写法
```

## 风险与备忘

- 扩散骨架若长期卡在 70-80% 踩音率，考虑在骨架头加 onset 检测辅助损失（audio onset 通道已存于 mel 里，可提峰值做粗标签）
- 站点分析器版本若更新（indexVersion 变化），CAL 表和刻度经验需重扫
- 用户敏感点排序不变：**对音 > 键型手感 > 难度分配 > 排列新颖度**
