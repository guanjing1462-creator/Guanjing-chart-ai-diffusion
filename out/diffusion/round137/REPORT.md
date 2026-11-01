# 第 137 轮：把管线修对（监督三个子代理的产出）

## 一、这轮做了什么
上一轮发现「按指定难度写」的实测结果可能被管线 bug 污染，这轮全部用来**验证并修复管线**，不新增训练。
结论：**三个 bug 都是真的，已有数字必须重测**。修复后重新标定，难度控制仍然成立且更干净。

---

## 二、修复一：Fourier 条件基底丢失（影响最大）
训练侧 `freq` 是 `requires_grad=False` 的 Parameter（在 state_dict 里），
旧采样器把它当普通 Tensor 建，strict 加载失败后**静默退化成 strict=False**，
于是丢掉 checkpoint 的基底、改用随机初始化。

**实测影响**（同输入、同权重、只换加载时的 RNG 种子）：

| 权重 | 训练 vs 旧推理 最大绝对误差 | 换种子后仍差 |
|---|---|---|
| cont | **0.2264** | 0.1422 |
| bal | **0.1604** | 0.0506 |

我的独立核验：checkpoint 内 `freq` 与随机初始化值**均值差 4.02**。

**修复**：`freq` 改为同名持久 buffer；只尝试一次 `strict=True`；缺 freq / 未知架构 / 结构矛盾 / NaN-Inf 一律明确报错，禁止静默回退。
**验收**：现有 cont/bal 权重可原样严格恢复，**同种子两次采样逐位一致（最大差异 0.000000）**。

## 三、修复二：另外三个不一致
1. **连续条件被截断**：NPZ 元数据用 `np.int64(diff)`，把 0.3/0.7 都记成 0（训练条件本身没问题，是记录错）；
2. **音频条件错位**：`C` 先构造，`X/S/D` 随机抽取后 `C` 没同步重排；且音频默认窗长 128 而 1/32 数据是 256；
3. **定向工具吞错误**：把「strict 加载失败 / 超时 / 量表上界截断」都笼统当成「低于下限」，会掩盖真实故障。

**验收**：`tests/test_diffusion32_consistency.py` **54 项全部通过**（含真实旧权重 strict 恢复、三种目标函数、梯度、条件对齐、损坏结构拒绝、CLI 传递、量表截断识别）。

## 四、修复三：训练/验证泄漏
数据审计确认三件事：
- 现有训练是 `torch.randperm(全部窗口)` 再切分 → **同一首歌的重叠窗口同时进训练和验证**；
- 旧 `timing()` 解析：**378 张谱的首个 timing 候选取自 [TimingPoints] 之外**；**2567 张谱的红线计数判错**；
- 旧三份 dataset **只有 X/style/diff，没有来源**，无法证明无泄漏。

**重建结果**（`out/diffusion/round136-data/` + `out/diff-data-grouped/dataset.npz`）：

| 项 | 值 |
|---|---|
| 原文件 | 2617 → 拒收 16（同 offset 多红线冲突）→ 2601 |
| 去重 | 30（19 字节重复 + 11 玩法重复）→ **2571 唯一谱** |
| 同源组 | **175 组** |
| 划分 | train **2299** / val **148** / test **124** 谱（组 141/17/17） |
| 窗口 | **95574**（[4,256]，stride 128） |
| **跨集交集** | chart_id 0 / group_id 0 / 共享谱窗口 0（三对划分全部为 0）✅ |
| 占位率 / p132 | 0.1986 / 0.2637（人类 p132 0.300） |

诚实边界：`window_overlap_audited=False`、`existing_npz_no_leakage_proven=False`——旧 NPZ 依然不能证明安全，新数据集才可用；2601 张谱**全部缺实际音频**，分组只覆盖已观测的元数据/引用/内容身份。

## 五、重新标定难度曲线（在修复后的基线上）
用修好的采样器重测（cont 权重，style=1，4 窗口，阈值 0.2，200 BPM）：

| 条件 | Koishi 难度 | 判定 |
|---|---|---|
| 0.10 | **12.62** | jack |
| 0.30 | **13.65** | stamina |
| 0.50 | **14.95** | jack |
| 0.70 | **16.51** | jack |
| 0.90 | 超出量表（>17） | jack |

**单调递增，覆盖 12.62–16.51**，比旧曲线（12.31–16.36 且中段挤压）更舒展。
数据存 `out/diffusion/curve-cont.json`，脚本 `tools/calc-diff-curve.py`（可 `--repeat` 复跑）。

**旧曲线作废**：第 133/134 轮那条 12.31→16.36 是在错误基底上量的。

## 六、端到端验收
```
python3 tools/target-diffusion.py --ckpt out/models/diffusion32-cont.pt --target 15 --style 1 --num 4 -o /tmp/tgt15-new.osu
第 1 轮：条件 0.506 → Koishi 15.13（目标 15.0）✅
```
工具已改为默认使用新实测曲线，并在输出里主动打印「单次命中不能证明精确控难」的警告。

## 七、下一步（明确）
1. **用 `out/diff-data-grouped/dataset.npz`（按组划分）重训**——这是第一份可信的训练集；
2. 训练/验证都按 `split` 键取，训练器要接 `chart_id/group_id/split`；
3. 重训后再重标曲线，用 val 上的实测难度而非训练损失验收；
4. speed/stamina 的键型分化仍需解决（上轮均衡采样只是修好 speed、弄弱 stamina）。

## 交付
- 修复：`tools/{train-diffusion32c,train-diffusion32b,sample-diffusion32c,target-diffusion}.py`
- 测试：`tests/test_diffusion32_consistency.py`（54 项通过）、`tests/test_gj_data_manifest.py`（18 项通过）
- 新数据：`out/diff-data-grouped/dataset.npz`（95574 窗口，按组划分）
- 标定：`out/diffusion/curve-cont.json`、`tools/calc-diff-curve.py`
- 报告：`out/diffusion/round136-consistency/REPORT.md`、`out/diffusion/round136-data/REPORT.md`
