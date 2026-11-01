#!/usr/bin/env python3
"""按目标 Koishi 难度出谱：给定目标定数，自动搜索连续难度条件并验证。

流程：以修复后实测曲线估起点 → 采样 → 转谱 → Koishi 测难度 → 启发式修正。
默认曲线仅限 cont/style1/4窗口/阈值0.2/200BPM 的单次测量，不是精确控难证明。
退出码：0=本次候选误差≤0.5，1=评测/采样失败，2=未达标（含量表上界截断）。

用法：
  python3 tools/target-diffusion.py --ckpt out/models/diffusion32-cont.pt --target 14 --style 1 -o out/tgt14.osu
"""
import argparse, glob, json, math, os, shutil, subprocess, sys, tempfile

# 修复后单次测量：out/diffusion/curve-cont.json（cont/style1/4窗口/0.2/200BPM）。
# 0.9 的 tier 为 '> Emik Zeta high'：17.2 只是 >17 截断值的搜索代理，非实测定数。
CURVE = [(0.1, 12.62), (0.3, 13.65), (0.5, 14.95), (0.7, 16.51), (0.9, 17.2)]
# 第 133 轮旧曲线受 Fourier bug 影响，只允许显式回退，不能用于有效控难结论。
LEGACY_CURVE = [(0.1, 12.31), (0.3, 14.30), (0.5, 15.48), (0.7, 15.96), (0.9, 16.36)]


def cond_for(target, use_unverified=False):
    pts = sorted(LEGACY_CURVE if use_unverified else CURVE)
    if target <= pts[0][1]:
        return pts[0][0]
    if target >= pts[-1][1]:
        return pts[-1][0]
    for (c0, d0), (c1, d1) in zip(pts, pts[1:]):
        if d0 <= target <= d1:
            return round(c0 + (target - d0) * (c1 - c0) / max(1e-6, d1 - d0), 3)
    return pts[len(pts) // 2][0]


def sample(ckpt, style, cond, num, thr, outdir, bpm, pred=None, seed=0):
    cmd = [sys.executable, 'tools/sample-diffusion32c.py', '--ckpt', ckpt,
           '--style', str(style), '--diff', str(cond), '--num', str(num),
           '--threads', '1', '--quiet', '--threshold', str(thr), '--outdir', outdir,
           '--bpm', str(bpm), '--seed', str(seed)]
    if pred is not None:
        cmd += ['--pred', pred]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'采样失败（exit {result.returncode}）：{result.stderr or result.stdout}')
    rolls = sorted(glob.glob(os.path.join(outdir, 'roll-*.npy')))
    if not rolls:
        raise RuntimeError('采样器未产生窗口，拒绝继续测难度')
    out = os.path.join(outdir, 'chart.osu')
    cmd2 = [sys.executable, 'tools/roll-to-osu32.py'] + rolls + ['--concat', '--bpm', str(bpm), '-o', out]
    result = subprocess.run(cmd2, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'转换失败（exit {result.returncode}）：{result.stderr or result.stdout}')
    return out


def koishi(path):
    r = subprocess.run(['node', 'tools/gj/batch-koishi.mjs', path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'Koishi 进程失败（exit {r.returncode}）：{r.stderr or r.stdout}')
    try:
        d = json.loads(r.stdout)[0]
        diff = d.get('diff')
        if d.get('error'):
            raise ValueError(f'分析错误：{d}')
        if diff is None and isinstance(d.get('tier'), str) and d['tier'].strip().startswith('>'):
            return None, d.get('pattern')  # 确定高于量表；不是超时/采样失败，也不是精确分数
        if isinstance(diff, bool) or not isinstance(diff, (int, float)) or not math.isfinite(diff):
            raise ValueError(f'没有有效难度或明确的上界截断信息：{d}')
        return float(diff), d.get('pattern')
    except (ValueError, TypeError, IndexError, AttributeError, KeyError) as e:
        raise RuntimeError(f'Koishi 结果不可用于评测：{e}') from e


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='out/models/diffusion32-cont.pt')
    ap.add_argument('--target', type=float, required=True, help='目标 Koishi 难度（12-16）')
    ap.add_argument('--style', type=int, default=1)
    ap.add_argument('--num', type=int, default=4, help='拼接窗口数（越多谱越长越准）')
    ap.add_argument('--thr', type=float, default=0.2, help='解码阈值（按占位率标定）')
    ap.add_argument('--bpm', type=float, default=200.0)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--max-rounds', type=int, default=3)
    ap.add_argument('--pred', choices=['eps', 'x0', 'v'], default=None,
                    help='传给采样器；默认恢复 checkpoint 元数据，禁止误标覆盖')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--initial-cond', type=float, default=None, help='覆盖曲线给出的搜索起点')
    ap.add_argument('--use-unverified-curve', action='store_true',
                    help='显式回退到涉及 Fourier bug 的第133轮旧曲线（仅供诊断）')
    a = ap.parse_args(argv)
    if (not math.isfinite(a.target) or a.max_rounds < 1
            or (a.initial_cond is not None and not 0 <= a.initial_cond <= 1)):
        ap.error('target 必须有限，initial-cond 必须在 [0,1]，max-rounds 必须 >= 1')
    if a.use_unverified_curve:
        print('[警告] 显式回退旧曲线：涉及 Fourier 加载 bug，不可作为有效标定。', file=sys.stderr)
    print('[注意] 默认曲线仅来自修复后 cont/style1/4窗口/阈值0.2/200BPM 的单次测量；'
          '单种子单调/单次命中不能证明精确控难。', file=sys.stderr)
    cond = a.initial_cond if a.initial_cond is not None else cond_for(a.target, use_unverified=a.use_unverified_curve)
    best = None
    for rnd in range(1, a.max_rounds + 1):
        tmp = tempfile.mkdtemp(prefix='tgt_')
        try:
            path = sample(a.ckpt, a.style, cond, a.num, a.thr, tmp, a.bpm,
                          pred=a.pred, seed=a.seed)
            diff, pat = koishi(path)
        except RuntimeError as e:
            print(f'第 {rnd} 轮失败：{e}', file=sys.stderr)
            return 1
        above_scale = diff is None
        score = 17.2 if above_scale else diff
        shown = '>17（量表截断；搜索代理17.2）' if above_scale else diff
        print(f'第 {rnd} 轮：难度条件 {cond:.3f} → Koishi {shown}（目标 {a.target}，判定 {pat}）')
        if not above_scale:
            if best is None or abs(diff - a.target) < abs(best[0] - a.target):
                best = (diff, cond, path, pat)
            if abs(diff - a.target) <= 0.5:
                shutil.copy(path, a.out)
                print(f'  ✅ 达标 → {a.out}')
                return 0
        # 未验证的启发式步长；截断代理仅用于降低条件，不能冒充精确命中。
        slope = 5.0 if cond < 0.4 else 2.0
        correction = (a.target - score) / slope
        if above_scale:
            correction = min(correction, -0.05)
        cond = max(0.05, min(0.9, cond + correction))
    if best is None:
        print('未达标：所有候选均高于 Koishi 量表上限，未将截断代理当作精确难度。', file=sys.stderr)
        return 2
    diff, cond, path, pat = best
    shutil.copy(path, a.out)
    print(f'未达标，仅保存最接近候选：条件 {cond:.3f} → {diff}（目标 {a.target}，差 {abs(diff - a.target):.2f}）→ {a.out}')
    return 2


if __name__ == '__main__':
    sys.exit(main())
