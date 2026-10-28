#!/usr/bin/env node
/* 给定 BPM，从 onset 峰反解「第一拍在哪」，让音符能落在音乐的小节/分音网格上。
 *
 *   node tools/calibrate-grid.mjs analysis.json --bpm 185
 *
 * 为什么要单独做这一步：
 *   之前用自相关自动测 BPM，这首测出 93.8（置信度 16%），实际是 185 的倍频错。
 *   BPM 一错，小节线全歪，音符虽然"贴着 onset"却不在拍子上——打起来是散的。
 *   这里不猜 BPM（人来给），只解相位：在 [0, beat) 里搜一个 offset，
 *   让最多的强 onset 落在 1/4 拍网格上；再在 4 个相位里挑重拍作为小节起点。
 */
import fs from 'node:fs';

const args = process.argv.slice(2);
const file = args.find((a) => !a.startsWith('-'));
const get = (k, d) => {
  const i = args.indexOf(`--${k}`);
  return i >= 0 ? args[i + 1] : d;
};
if (!file) {
  console.error('用法: node tools/calibrate-grid.mjs analysis.json --bpm 185 [--sigma 30] [--sub 16]');
  process.exit(1);
}

const src = JSON.parse(fs.readFileSync(file, 'utf8'));
const bpm = parseFloat(get('bpm', '185'));
const sigma = parseFloat(get('sigma', '30'));     // 判定"落在网格上"的容差
const sub = parseInt(get('sub', '16'), 10);       // 分音：16 = 1/16 拍
const beat = 60000 / bpm;

const peaks = src.peaks;
const energy = src.fine.energy;
const hop = src.fine.hopMs;
const ev = peaks.map((p) => ({ t: p * hop, e: energy[p] })).filter((x) => x.e > 0);
const totalE = ev.reduce((a, x) => a + x.e, 0);

/* 高斯核打分：峰离网格越近、能量越大，贡献越大。
   用高斯而不是硬阈值，是为了让"差 3ms"和"差 28ms"有区别，
   相位才能收敛到亚毫秒级，而不是在几个候选之间跳。 */
function score(offset, period) {
  let s = 0;
  const k = 1 / (2 * sigma * sigma);
  for (const x of ev) {
    let d = (x.t - offset) % period;
    if (d < 0) d += period;
    if (d > period / 2) d = period - d;
    s += x.e * Math.exp(-d * d * k);
  }
  return s;
}

// ---- 1. 拍相位：offset ∈ [0, beat) ----
let best = { off: 0, s: -1 };
for (let o = 0; o < beat; o += 0.5) {
  const s = score(o, beat);
  if (s > best.s) best = { off: o, s };
}
// 粗搜后细搜 ±2ms
for (let o = best.off - 2; o <= best.off + 2; o += 0.05) {
  const s = score(o, beat);
  if (s > best.s) best = { off: o, s };
}
const beatOff = best.off;

// ---- 2. 小节起点：4 个相位里挑重拍 ----
let bar = { idx: 0, s: -1 };
for (let j = 0; j < 4; j++) {
  const o = beatOff + j * beat;
  const s = score(o, beat * 4);
  if (s > bar.s) bar = { idx: j, s };
}
const barOff = beatOff + bar.idx * beat;

// ---- 3. 报告落网率 ----
const grid = beat / (sub / 4);   // 1/16 拍的毫秒数 = beat/4
function hitRate(offset, period, tol) {
  let hit = 0, all = 0;
  for (const x of ev) {
    let d = (x.t - offset) % period;
    if (d < 0) d += period;
    if (d > period / 2) d = period - d;
    all += x.e;
    if (d <= tol) hit += x.e;
  }
  return hit / (all || 1);
}

const r = {
  bpm,
  beatMs: +beat.toFixed(3),
  sub, gridMs: +grid.toFixed(3),
  offsetMs: +barOff.toFixed(1),
  beatOffsetMs: +beatOff.toFixed(1),
  downbeatInBar: bar.idx,
  snap: {
    quarter: +(hitRate(barOff, beat, 25) * 100).toFixed(1),
    eighth: +(hitRate(barOff, beat / 2, 20) * 100).toFixed(1),
    sixteenth: +(hitRate(barOff, grid, 15) * 100).toFixed(1),
  },
  bars: Math.floor((src.durationSec * 1000 - barOff) / (beat * 4)),
};

console.log(JSON.stringify(r));
