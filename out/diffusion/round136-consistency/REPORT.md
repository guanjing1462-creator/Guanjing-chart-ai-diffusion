# Round136 · 连续扩散训练/推理一致性审计

## 结论与范围

已修复连续采样器丢弃训练 Fourier 基底、非严格加载掩盖结构问题、连续条件存盘截断、预测目标缺省猜测以及音频条件与窗口重排不一致。现有 [cont 权重](/root/work/yamero-ai/proj/out/models/diffusion32-cont.pt) 与 [bal 权重](/root/work/yamero-ai/proj/out/models/diffusion32-bal.pt) 都保留完整 `freq`，可原样 strict 恢复，**不需要为修复频率重新训练，也没有改写原 checkpoint**。

本代理未运行真实数据长训练、未生成谱面/谱包，未修改离散 [sample-diffusion32.py](/root/work/yamero-ai/proj/tools/sample-diffusion32.py)。本地回归只使用 CPU、`torch.set_num_threads(1)`、小型合成张量/临时 checkpoint；涉及采样 CLI、转谱及 Koishi 的测试使用 mock，不产生输出谱面。

**重要：本轮证明模型恢复与公式一致，不证明已学到可靠的键型分化或精确难度控制。**

## 1. 修复前的可复现问题

### 1.1 Fourier 条件基底丢失

训练侧 `freq` 是 `requires_grad=False` 的 Parameter，包含于 `state_dict`；旧连续采样器却将它作为普通 Tensor。strict 加载因 unexpected `freq` 失败后，又自动尝试 `strict=False`，因此忽略 checkpoint 的基底并使用随机初始化频率。

修复前实测：输入由 `torch.Generator().manual_seed(136)` 产生，`x.shape=[2,4,32]`、`t=[0,199]`、`style=[0,1]`、连续条件 `[0.3,0.7]`；训练 c 严格恢复 checkpoint，采样器分别以初始化种子 11、22 加载同一文件。每次旧加载均提示 `unexpected=['freq']`。

| checkpoint | 加载初始化种子 | 训练/旧推理 forward 最大绝对误差 | 平均绝对误差 |
|---|---:|---:|---:|
| cont | 11 | 0.2264118493 | 0.0590696074 |
| cont | 22 | 0.1398747563 | 0.0341222771 |
| bal | 11 | 0.1604394317 | 0.0440325737 |
| bal | 22 | 0.1138448417 | 0.0347035080 |

只改变模型加载时的随机种子，同输入输出的最大差异仍达 cont **0.1421951354**、bal **0.0505589545**，不是采样噪声造成的差别。

监督方另提供的核验事实（非本代理额外运行）：checkpoint 内 `freq` 与随机初始化基底的“均值差 4.02”。该消息未附初始化种子及详细统计定义；本报告保留其来源，不把它冒充本地独立重测值。

### 1.2 其他不一致

- 连续条件实际传入 forward 时已是 float；问题在 NPZ 元数据使用 `np.int64(diff)`，把 0.3/0.7 均记为 0，不是训练条件本身被离散化。
- 缺少 `pred` 的 checkpoint 原先默认 eps，可能错误解释 x0/v 模型输出；显式误标也没有端到端校验。
- 音频 `C` 先构造，`X/S/D` 随机抽取/重排后 `C` 未同步重排；此外音频构造默认窗长 128，而 1/32 数据窗长为 256。
- 定向工具原先吞掉子进程错误，并把没有数值的 Koishi 结果笼统解释为低于下限；这会掩盖 strict 加载失败、超时及量表上界截断。

## 2. 最小代码改动

| 文件 | 改动 |
|---|---|
| [train-diffusion32c.py](/root/work/yamero-ai/proj/tools/train-diffusion32c.py) | `freq` 改为同名持久 buffer，保持旧 state 键兼容；非法 target 显式拒绝；音频传入实际窗长且 `C=C[order]`；新增模型类型、占位率归一化 lo/hi、验证源行号元数据。 |
| [train-diffusion32b.py](/root/work/yamero-ai/proj/tools/train-diffusion32b.py) | 同上；保留 b 的风格平衡逻辑，额外记录 `style_balance`，没有整段覆盖架构。 |
| [sample-diffusion32c.py](/root/work/yamero-ai/proj/tools/sample-diffusion32c.py) | 同名持久 `freq`；只尝试一次 `strict=True`；拒绝缺频率、核心键缺失/多余/形状冲突、非浮点或 NaN/Inf 权重、未知模型类型、结构元数据矛盾、非法预测目标；恢复 `prediction_type`，缺 pred 时要求显式 `--pred`；拒绝冲突覆盖；共享反解公式；diff 元数据改 float64；`--cond-zero` 确实覆盖条件文件并正确记录；默认 CPU 单线程。 |
| [target-diffusion.py](/root/work/yamero-ai/proj/tools/target-diffusion.py) | `--pred`/seed/BPM 显式传递，保留原始错误；默认采用监督方新测曲线，旧曲线仅显式回退；区分有效上界截断与失败；截断只用作搜索代理，不能假报命中；0/1/2 分别代表本次容差内命中/执行失败/未达标。 |
| [test_diffusion32_consistency.py](/root/work/yamero-ai/proj/tests/test_diffusion32_consistency.py) | 54 项标准库 unittest，覆盖真实旧权重、两训练器三种目标、梯度、条件对齐、损坏结构、元数据、CLI 传递及量表截断。 |

旧 Parameter 与新 buffer 的 state 键和 Tensor 均为 `freq`，因此不丢权重、不新增随机补齐、不需要转换 checkpoint。`model.parameters()` 统计不再包含固定频率的 64 个元素，这是改为非训练 buffer 的预期统计变化，不是删除神经网络层。

对于结构完整但没有 pred 元数据的旧 raw state，显式 `--pred eps/x0/v` 可以恢复；原有 cont/bal 文件有 `pred=x0`，无需覆盖。已存 pred 与 `--pred` 不一致直接失败，不允许用覆盖掩盖误标。

## 3. 回归命令与结果

在项目根目录运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m unittest discover -s tests -p 'test_diffusion32_consistency.py' -v
```

最终结果：**54/54 通过，0 failures、0 errors、0 skips**；unittest 实际执行 **10.659 秒**，包括解释器/torch 导入的外层总耗时 **22.948 秒**，exit code 0。完整输出见 [TEST-final.log](/root/work/yamero-ai/proj/out/diffusion/round136-consistency/TEST-final.log)。首次运行 48 项时有一项测试读取 `mock_open.call_args` 的自有 mock 错误（非模型失败）；修正为读取 mock 返回句柄后通过，首次记录保留在 [TEST.log](/root/work/yamero-ai/proj/out/diffusion/round136-consistency/TEST.log)。

覆盖内容：

1. 两个真实 checkpoint × 两个训练器，严格恢复所有核心权重和频率；初始化种子 11/22 不影响已恢复模型输出。
2. 修复后真实窗长 `[2,4,256]` 下，训练 b/c 与推理 forward **逐位一致，最大绝对误差 0**。此验收使用实际 256 格窗口；修复前诊断使用上文的 32 格输入，不混称为同一批统计量。
3. `eps/x0/v` × b/c 的独立目标公式与反解：覆盖 `t=0,1,99,199`，绝对容差 `2e-6`、相对容差 `1e-6`；每种模式各做一个真实 SGD 步，梯度有限、核心权重改变、频率不变，恢复后 forward 逐位相等。
4. 每个训练器各做一次仅 8 个合成窗口、随机取 4、1 epoch 的小型集成测试，检查训练/验证音频来源、256 格条件及 checkpoint 元数据；没有真实数据训练。
5. 20 项结构/内容/元数据损坏变体、raw 旧 state 恢复、strict-only 尝试、真实 CLI 错误退出，均禁止静默 fallback。
6. 0.37 的连续条件在 DDPM、NPZ/JSON 与定向命令中保留；同 seed 内存短采样逐位一致；`--pred` 四种情况（eps/x0/v/由 checkpoint 读取）逐级验证。
7. 量表上界 `diff=null,tier='> Emik Zeta high',error=null` 被识别为截断，不同于 timeout/无效 JSON/进程失败；17.2 代理不会直接算作命中。

实际模型的连续条件敏感性（相同 x/t/style，非谱面难度测量）：

| checkpoint | 条件 0.30→0.31 的 forward 最大变化 | 条件 0.30→0.70 的最大变化 | 不同加载种子的变化 |
|---|---:|---:|---:|
| cont | 0.00850930810 | 0.295649052 | 0 |
| bal | 0.00911349058 | 0.338927627 | 0 |

b/c 两训练定义均得到相同结果。**这些非零变化只证明连续条件进入模型、没有被取整，不证明实际 Koishi 难度单调，更不证明定数可精确指定。**

监督方额外验收：现有 cont checkpoint 可正常采样；相同 seed 两次采样实测逐位一致，最大差异 **0.000000**。该真实采样由监督方运行；本代理不重复生成谱面。

## 4. 修复后的单次曲线与定向工具

来源为监督方提供、已读取的 [curve-cont.json](/root/work/yamero-ai/proj/out/diffusion/curve-cont.json)，生成工具为监督方新增的 [calc-diff-curve.py](/root/work/yamero-ai/proj/tools/calc-diff-curve.py)。本代理未执行该标定脚本。

测量配置：cont checkpoint、style=1、4 个窗口、阈值 0.2、200 BPM，每个条件只测 1 次。

| 连续条件 | 实际 Koishi 结果 | 搜索用途 |
|---:|---|---|
| 0.10 | 12.62 | 实测插值点 |
| 0.30 | 13.65 | 实测插值点 |
| 0.50 | 14.95 | 实测插值点 |
| 0.70 | 16.51 | 实测插值点 |
| 0.90 | null，tier 为 `> Emik Zeta high`（>17） | 只使用 17.2 搜索代理；不是实测 17.2 |

有限点在这一次配置下递增、实测覆盖 **12.62–16.51**。旧第133轮曲线受错误 Fourier 基底影响，作废为有效标定证据；默认已换成新曲线，`--use-unverified-curve` 仅供显式诊断回退，`--initial-cond` 可覆盖起点。

> 单种子、每点一次的单调性或单次命中，不能证明精确控难。新曲线是特定配置的搜索起点，不能直接推广到 bal checkpoint、其他 style/阈值/BPM/长度。上界截断不是采样失败，也不能把搜索代理冒充实际分数。

另外，监督方标定脚本的 `--repeat` 循环当前未改变采样 seed；仅提高 repeat 会重复同一确定性样本，不能当作独立多种子验证。后续验证应显式设计不同 seed；本轮按监督方要求不再修改该工具。

## 5. 泄漏边界与下一次训练的硬要求

已阅读其他代理交付的 [数据报告](/root/work/yamero-ai/proj/out/diffusion/round136-data/REPORT.md) 和 [manifest](/root/work/yamero-ai/proj/out/diffusion/round136-data/manifest.json)：唯一谱 **2571**、同源组 **175**；train/val/test 为 **2299/148/124** 张（141/17/17 组）；已观测 group/chart/玩法 hash/原文件 hash/身份 key 的跨集交集均为 **0**。该保证限于已观测身份的传递闭包，不是未识别音频的绝对同曲证明。

**在有泄漏风险的旧窗口随机验证集上得到的 val 0.0811 等数字不可信，不能当成无泄漏泛化成绩；下一次训练必须使用已交付的同源分组划分。** 本次已检查的旧 cont/bal checkpoint 内 val 分别为 0.0829923153、0.0814402103，同样不能据此声称独立验证有效，也不能把监督方提到的 0.0811 冒称为这两个文件内的精确字段。

当前两训练器仍保留原窗口 `randperm` 后划 train/val 的流程；本轮一致性补丁**没有接入新 manifest，也没有消除训练验证泄漏**。后续必须先按 source/group 固定 train/val/test，再切窗并让每个窗口继承 split；占位率归一化只在 train 拟合。当前归一化仍在旧全量数据上计算，虽已保存 lo/hi，但保存元数据本身不消除泄漏。`val_source_idx` 只是旧 NPZ 的行号，不是 source_id/group_id，不是无泄漏证书。

## 6. 尚未解决/不能推出的结论

- 没有独立多种子难度分布、误差区间、按风格/阈值/长度的交叉校准；没有评估键型质量或确认风格标签能稳定分化成真实键型。
- 旧 checkpoint 没有占位率归一化参数，不能仅凭权重还原原训练数据的 5%/95% 数值；本轮只保证给定同一条件值时的网络函数一致。
- strict state 校验能发现键、形状、显式元数据矛盾，不能从同形状权重数学推断被人伪造但仍合法的 eps/x0/v 标签或历史 beta schedule；原两文件的已存 x0 与现有训练实现一致。
- 支持现有 4K 连续 Fourier 结构与线性 beta schedule；没有为离散模型、未知结构或缺失 Fourier 基底发明补救权重。缺 `freq` 的旧权重无法精确恢复，必须明确拒绝，不能随机补齐。
- x0 模式现有 DDPM 的 `[-1,2]` 裁剪保留；独立重建公式测试不裁剪，以验证代数本身，不将其混同为生成质量/物理精度证明。
- 本轮恢复模型一致性不修复旧二值数据中的 LN 时值缺失、变速/网格解析和来源证明问题；这些事项以数据代理报告为准，后续训练前需完成数据重建与分组接入。
