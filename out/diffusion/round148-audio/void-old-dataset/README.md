# 作废（VOID）：旧数据集 `out/diff-data-audio-big` 上的训练产物

旧集的音频条件 `C` 比音符系统性晚约一格（≈40ms）：δ=0 时逐谱 X–C 相关中位 −0.062，
δ=+50ms 才升到 +0.498（见 `out/diffusion/round149-audio-fixed/REPORT.md`）。

因此这里的所有训练日志/权重都**不可用于任何结论**：
- `A-train.*`：旧集 A 档（cond_dim=0，val 0.0946，15 epoch 中跑到 9 轮后按预算停）
- `B-train.*`：旧集 B 档（cond_dim=32，第 1 轮被中止）

有效产物见上级目录的 `A-fixed-train.*` / `B-fixed-train.*` 与 `REPORT.md`。
