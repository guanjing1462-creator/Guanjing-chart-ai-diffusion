#!/usr/bin/env node
/* 从 analysis.json 生成一张谱面的音符表（不写 .osu，交给 chartgen.py 拟合难度）。
 *
 *   node tools/gen-notes.mjs analysis.json --nps 11.4 --seed 7 -o notes.json
 *
 * 为什么单独一步：MSD 拟合要迭代好几轮，每轮只变 nps；
 * 把「音频 → 事件表」和「事件表 → 列分配」拆开太啰嗦，
 * 这里整轮重生成（一次 0.3 秒），保证同一 seed 下结果可复现。
 */
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const CP = require('../src/chart-patterns.js');

function parseArgs(argv) {
  const o = { pos: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const isFlag = a.startsWith('--') || (a.length === 2 && a.startsWith('-'));
    if (isFlag) {
      const k = a.replace(/^--?/, '');
      const v = argv[i + 1];
      if (v && !v.startsWith('-')) { o[k] = v; i++; } else { o[k] = true; }
    } else o.pos.push(a);
  }
  return o;
}

const args = parseArgs(process.argv.slice(2));
const analysisPath = args.pos[0];
if (!analysisPath) {
  console.error('用法: node tools/gen-notes.mjs <analysis.json> --nps 11.4 [--seed 7] [--keys 4] [-o notes.json]');
  process.exit(1);
}

const src = JSON.parse(fs.readFileSync(analysisPath, 'utf8'));
const nps = parseFloat(args.nps || '12');
const seed = parseInt(args.seed || '42', 10);
const keys = parseInt(args.keys || '4', 10);

// 有种子随机：同一 (nps, seed) 结果完全一致，拟合过程才可复现
let s = seed >>> 0;
const rnd = () => {
  s = (s * 1664525 + 1013904223) >>> 0;
  return s / 4294967296;
};

/* 键型配比按人类 tech 标杆（5898623/5442650，站点 Koishi 实测
   tech 概率 0.75/0.99、分类器特征 chord1Ratio 0.67/0.69）反推：
   人类 tech 图是**单音流主体**（67% 单音 + 26% 双押 + 7% 三押，
   跨手四列轮转），双押家族只做点缀。rotstream 主力（单音流+
   轮换点缀），stream 补纯单音段，chordjack 限比（cj 段不超过一成半，
   占比再高会被 Koishi 判 jack 主导——jackRatio 人类 0.09），
   breather 不进池（由呼吸段注入）。
   人类 tech 标杆七维（msd-full 实测）：stream 29.1 / js 30.2 /
   hs 27.4 / stamina 29.8 / jack 18.6 / cj 25.8 / tech 29.3。 */
const DEFAULT_W = {
  rotstream: 7.0, stream: 3.5, chordjack: 2.0, jumpstream: 2.0,
  handstream: 1.0, trill: 1.0, roll: 1.0, ladder: 1.0, staircase: 0.8,
};
let weights = DEFAULT_W;
if (args.weights) {
  // 部分覆盖：只改传入的键，其余保持默认
  const over = {};
  for (const kv of String(args.weights).split(',')) {
    const [k, v] = kv.split('=');
    if (k && v) over[k.trim()] = parseFloat(v);
  }
  if (Object.keys(over).length) weights = Object.assign({}, DEFAULT_W, over);
}

/* 和弦率：站点 Koishi 分类器实测（feat dump）——人类 tech 图
   （5898623/5442650）chord1Ratio 0.67/0.69、chord2Ratio 0.26/0.23、
   chord3Ratio 0.07/0.08，即 tap 层 chordRate ≈ 1.40：单音流主体、
   双押三押点缀。高密度段靠行频（1/16 格覆盖）而不是和弦加宽。
   旧版按 12 张 MSD 29-33 均值定 2.00——那批含 jack 型图（4010375
   chordRate 2.66），照抄它直接喂出 jackRatio 0.39 → jack 判定。 */
function autoChord(nps) {
  if (nps < 6) return 1.20;
  if (nps < 9) return 1.35;
  if (nps < 12) return 1.42;
  if (nps < 15) return 1.42;
  if (nps < 18) return 1.44;
  return 1.42;
}
const chordRate = args.chord != null ? parseFloat(args.chord) : autoChord(nps);

/* 网格模式：把 analysis 里的 accent（按 1/16 拍量出来的能量）传进去，
   让每个音符都落在分音上。--onsets 可以退回旧的「贴 onset 峰」模式做对比。 */
const useGrid = !args.onsets;

const r = CP.generateFromPeaks({
  keys,
  grid: useGrid ? {
    stepMs: src.stepMs,
    offsetMs: src.offsetMs,
    accent: src.accent,
    gridMin: args['grid-min'] != null ? parseFloat(args['grid-min']) : undefined,
  } : null,
  nps,
  weights,
  fine: src.fine,
  peaks: src.peaks,
  durationSec: src.durationSec,
  sectionSec: parseFloat(args.section || '8'),
  chordRate,
  dynExp: args.dyn != null ? parseFloat(args.dyn) : 1.6,
  segMulCap: args['seg-mul-cap'] != null ? parseFloat(args['seg-mul-cap']) : undefined,
  triRate: args['tri'] != null ? parseFloat(args['tri']) : undefined,
  dblRate: args['dbl'] != null ? parseFloat(args['dbl']) : undefined,
  breath: args.breath === 'false' ? false : true,
  rnd,
  rndSeed: seed,   // 让 generateFromPeaks 的配额标定循环可重放同种子随机数
});

const out = {
  notes: r.notes,
  meta: r.meta,
  sections: r.sections.length,
  bpm: src.bpm,
  offsetMs: src.offsetMs,
  keys,
};
const dest = args.o || 'notes.json';
fs.writeFileSync(dest, JSON.stringify(out));
console.log(JSON.stringify({
  out: dest, nps: r.meta.nps, chordRate, notes: r.meta.notes,
  events: r.meta.events, align: r.meta.alignRatio, sections: r.sections.length,
}));
