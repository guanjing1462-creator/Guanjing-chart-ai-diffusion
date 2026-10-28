/*!
 * chart-patterns.js — 4K 键型库（语义来自《4K 下落式音游齐民要术 v1.0》卷二）
 * 版本 1.0.0 · 零依赖
 *
 * 设计：每个键型是一个「步进器」——init 初始化状态，step 按第 i 步返回这一拍要按的列。
 * 生成器按段落（section）切换键型，模拟真实编排里「这一段是交互、下一段是乱」的写法。
 *
 * 文献定义对照（卷二 2.1.1 / 2.1.2）：
 *   minijack  子弹/短纵  小间距的连续单键
 *   anchor    锚键       同一列等间距反复出现，中间穿插其他列
 *   longjack  纵连       同一列连续等间距单键
 *   roll      滚动       同方向等间距，四列一组
 *   ladder    梯子       同方向排列的单键
 *   staircase 钩子       同方向排列后突然反向拐角
 *   trill     交互       两列等间距交替
 *   jumptrill 对拍       12/34 交替双押
 *   splittrill 对切      13/24 或 14/23 交替双押
 *   splitrail 散点       无规律单键
 *   stream    乱键       高速单键流
 *   jack      叠键       相邻行允许按键重叠
 *
 * 分度说明：文献里「16 分」= 每拍 4 个音符。本模块的 step 就是 1/16 拍，
 * 所以一个 16 分音符 = 1 step，8 分 = 2 step，4 分 = 4 step。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ChartPatterns = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  function randInt(n, rnd) { return Math.floor((rnd || Math.random)() * n); }

  /* 同种子 LCG：配额标定循环每轮要重造同一串随机数，
     否则 (nps, seed) 不可复现，拟合过程会漂。 */
  function lcg(seed) {
    var s = (seed == null ? 1 : seed) >>> 0;
    return function () {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  /* 排列偏好：文献 2.1.2.2 提到「脚排列 / 古典排列 / 现代排列」三种写法。
     站点 Koishi 的 alternationRatio（人类 tech 图 0.49/0.51，旧版生成器
     0.003——160 倍差距，直接判 jack）衡量左右手交替：单音流必须走
     「左-右-左-右」的跨手轮转，[0,1,2,3] 这种同侧两连会把交替率打没。
     三种排列全部为左右交替型。 */
  var ORDERS = {
    classic: [1, 2, 0, 3],
    modern: [0, 3, 1, 2],
    foot: [2, 0, 3, 1]
  };
  /* 跨手四列轮转排列池：单音层用。每个排列左右严格交替
     （0,1 = 左手，2,3 = 右手），段 init 时随机取一个。 */
  var ALT_ORDERS = [
    [0, 2, 1, 3], [0, 3, 1, 2], [1, 2, 0, 3], [1, 3, 0, 2],
    [2, 0, 3, 1], [2, 1, 3, 0], [3, 0, 2, 1], [3, 1, 2, 0]
  ];

  /* 手对轮换表：4 列两两组合共 6 种。人类 D14 tech 谱（5898623，Koishi
     pattern_rotation 0.53）的实测写法——双押列组合每拍都换（023→02→01→
     12→23→03→13），几乎不重复。固定手对连打（[01][01][01]）会被站点
     Koishi 判 jack（实测 repeat 0.616 / rotation 0.233）。
     顺序而非随机：轮换保证相邻两次必然不同，随机抽样有 1/6 概率撞车。 */
  var PAIRS = [[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2]];
  /* 三押轮换表：四种三列组合。同理，按序推进避免连续重复。 */
  var TRI = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

  /* ------------------------------------------------------------------ */
  /* 键型步进器                                                          */
  /* ------------------------------------------------------------------ */
  var PATTERNS = {
    /* 子弹：两连击同列，16 分 */
    minijack: {
      label: '子弹',
      init: function (s, rnd) { s.c = randInt(4, rnd); },
      step: function (s, i) { return [s.c]; },
      period: 2
    },

    /* 纵连：同列 4 连 */
    longjack: {
      label: '纵连',
      init: function (s, rnd) { s.c = randInt(4, rnd); },
      step: function (s, i) { return [s.c]; },
      period: 4
    },

    /* 锚键：一列锚定，其余列穿插 */
    anchor: {
      label: '锚键',
      // step 拿不到 rnd 参数，必须在 init 里把 rnd 存进 state——
      // 否则 randInt(2) 会落回 Math.random，同一 seed 生成的谱每次都不一样。
      init: function (s, rnd) { s.c = randInt(4, rnd); s.k = 0; s.rnd = rnd; },
      step: function (s, i) {
        if (i % 2 === 0) return [s.c];
        s.k = (s.k + 1 + randInt(2, s.rnd)) % 4;
        if (s.k === s.c) s.k = (s.k + 1) % 4;
        return [s.k];
      },
      period: 4
    },

    /* 滚动：四列单向（跨手交替序列——[0,1,2,3] 同侧两连会把站点
       Koishi 的 alternationRatio 打没，人类 tech 图 0.49） */
    roll: {
      label: '滚动',
      init: function (s, rnd) { s.dir = rnd() < 0.5 ? 1 : -1; s.seq = s.dir > 0 ? [0, 2, 1, 3] : [3, 1, 2, 0]; },
      step: function (s, i) { return [s.seq[i % 4]]; },
      period: 4
    },

    /* 梯子：同向排列，长度不定 */
    ladder: {
      label: '梯子',
      init: function (s, rnd) { s.dir = rnd() < 0.5 ? 1 : -1; s.pos = s.dir > 0 ? 0 : 3; },
      step: function (s, i) {
        var c = s.pos;
        s.pos += s.dir;
        if (s.pos > 3) { s.pos = 3; s.dir = -1; }
        if (s.pos < 0) { s.pos = 0; s.dir = 1; }
        return [c];
      },
      period: 4
    },

    /* 钩子：同向爬升后反向拐角 */
    staircase: {
      label: '钩子',
      init: function (s, rnd) { s.seq = rnd() < 0.5 ? [0, 1, 2, 3, 2, 1, 0, 1] : [3, 2, 1, 0, 1, 2, 3, 2]; },
      step: function (s, i) { return [s.seq[i % 8]]; },
      period: 8
    },

    /* 交互：两列交替 */
    trill: {
      label: '交互',
      init: function (s, rnd) {
        var a = randInt(4, rnd);
        var b = (a + 1 + randInt(2, rnd)) % 4;
        s.pair = [a, b];
      },
      step: function (s, i) { return [s.pair[i % 2]]; },
      period: 2
    },

    /* 对拍：12 / 34 双押交替 */
    jumptrill: {
      label: '对拍',
      init: function (s, rnd) { s.flip = rnd() < 0.5; },
      step: function (s, i) {
        var a = (i % 2 === 0) ? [0, 1] : [2, 3];
        return s.flip ? a.slice().reverse() : a;
      },
      period: 4
    },

    /* 对切：13 / 24 或 14 / 23 双押交替 */
    splittrill: {
      label: '对切',
      init: function (s, rnd) { s.pair = rnd() < 0.5 ? [[0, 2], [1, 3]] : [[0, 3], [1, 2]]; },
      step: function (s, i) { return s.pair[i % 2].slice(); },
      period: 4
    },

    /* 散点：无规律单键 */
    splitrail: {
      label: '散点',
      init: function (s, rnd) { s.last = -1; },
      step: function (s, i, rnd) {
        var c = randInt(4, rnd);
        if (c === s.last) c = (c + 1 + randInt(2, rnd)) % 4;
        s.last = c;
        return [c];
      },
      period: 1
    },

    /* 乱键：高速单键流，走排列顺序 */
    stream: {
      label: '乱键',
      init: function (s, rnd) {
        var keys = Object.keys(ORDERS);
        s.order = ORDERS[keys[randInt(keys.length, rnd)]];
        s.k = 0;
      },
      step: function (s, i) {
        var c = s.order[s.k % 4];
        s.k++;
        return [c];
      },
      period: 4
    },

    /* 叠键：相邻行重叠的和弦（同样受 1/8 拍间隔约束） */
    jack: {
      label: '叠键',
      init: function (s, rnd) { s.pair = rnd() < 0.5 ? [0, 2] : [0, 3]; s.alt = 0; s.lastT = -1e9; },
      step: function (s, i, rnd, evt) {
        var t = evt ? evt.t : i * 80;
        if (t - s.lastT < 155) return [3 - (i % 4)];
        s.lastT = t;
        if (i % 4 === 3) { s.alt = (s.alt + 1) % 3; }
        return s.pair.slice();
      },
      period: 4
    },

  /* 轮换双押流（主力键型 v3）：站点 Koishi 分类器实测口径——
     人类 tech 图（5898623/5442650）chord1Ratio 0.67/0.69、双押 0.26、
     三押 0.07、alternationRatio 0.49、rhythmEntropy 2.47。旧版双押
     主体（chord2Ratio 0.53）喂出 jackRatio 0.39 → jack 判定 0.996。
     v3 构成：拍级模式轮换（5 种节奏型每拍随机选——重音位双押/三押、
     其余位跨手单音轮转），单音 ~70% + 双押 ~25% + 三押 ~5%。
     拍级随机同时解决 rhythmEntropy（固定四列轮转只有 0.74）。 */
  rotstream: {
    label: '轮换双押流',
    init: function (s, rnd, ctx) {
      s.pi = randInt(6, rnd);          // 手对轮换游标
      s.ti = randInt(4, rnd);          // 三押轮换游标
      s.k = 0;                         // 单音层游标
      s.beatMode = 0;
      s.lastPair = [0, 1];
      s.load = (ctx && ctx.load) || null;   // 共享双手负载追踪器
      s.order = ALT_ORDERS[randInt(ALT_ORDERS.length, rnd)];
      /* 段能量高→双押三押点缀密（爆发段敢排），低能段退回单音流。 */
      var mul = (ctx && ctx.secMul) || 1;
      s.rich = Math.max(0, Math.min(1, (mul - 0.8) / 1.6));   // 0..1
    },
    step: function (s, i, rnd, evt) {
      var cell = evt && evt.cell != null ? evt.cell : i;
      /* 1/32 半格事件：纯单音轮转。不能走下面的 pos 逻辑——
         pos===0 的半格会重抽拍级模式（每个半格都重新掷，拍结构被打散），
         pos===2 的半格会撞进双押槽。人类 tech 图的 1/32 串就是
         跨手单音流，与主网格的节奏模式并行不悖。 */
      if (evt && evt.sub) return [s.order[s.k++ % 4]];
      var pos = cell & 3;
      /* 拍级模式：每 1/4 拍（4 格）重新抽。rich（段能量）高时偏向
         密模式（2/3），低时偏向疏模式（0/4）——爆发段重音位排双押，
         铺垫段纯单音流，这就是人类谱师"高潮前收一拍"的微观版。
         mode4（隔格休止）控制在 8%：音乐本身的休止由选格阶段听写，
         随机休止开到 20% 会把 tap 密度打到 11 nps（人类 tech 14.8）。 */
      if (pos === 0) {
        var r = rnd();
        var pRich = 0.25 + 0.45 * s.rich;          // 密模式（2/3）概率
        if (r < pRich * 0.45) s.beatMode = 3;       // 拍头三押+半拍双押
        else if (r < pRich) s.beatMode = 2;         // 拍头+半拍双押
        else if (r < pRich + 0.22) s.beatMode = 1;  // 拍头双押
        else if (r < pRich + 0.80) s.beatMode = 0;  // 全单音
        else s.beatMode = 4;                        // 隔格休止（8%）
      }
      function pairOut(p) {
        /* 阈值从 0.35 收到 0.12：负载计数是全曲累计的，0.35 条音符的
           偏差在 4000 音的图上根本触发不了——实测双手差 13%（人类 <5%）。
           现在 0.12 条就翻面，配合 ALT_ORDERS 的期望均衡压回人类带内。 */
        if (s.load && Math.abs(s.load.lh - s.load.rh) > 0.12) {
          var lc = p.filter(function (c) { return c < 2; }).length;
          if (s.load.lh - s.load.rh > 0.12 && lc === 2) p = [2, 3];
          if (s.load.rh - s.load.lh > 0.12 && lc === 0) p = [0, 1];
        }
        s.lastPair = p;
        return p;
      }
      switch (s.beatMode) {
        case 1: return pos === 0 ? pairOut(PAIRS[s.pi++ % PAIRS.length].slice())
                                 : [s.order[s.k++ % 4]];
        case 2: return pos % 2 === 0 ? pairOut(PAIRS[s.pi++ % PAIRS.length].slice())
                                     : [s.order[s.k++ % 4]];
        case 3:
          if (pos === 0) return pairOut(TRI[s.ti++ % TRI.length].slice());
          if (pos === 2) return pairOut(PAIRS[s.pi++ % PAIRS.length].slice());
          return [s.order[s.k++ % 4]];
        case 4: return pos === 3 ? [] : [s.order[s.k++ % 4]];
        default: return [s.order[s.k++ % 4]];
      }
    },
    period: 8,
    yield: 1
  },

  /* 疏流（呼吸段专用）：只在正拍与 1/8 位出单音，1/16 位休止。
     人类 D14 图 recovery_ge0.5 ≈ 0.02-0.03（站点 Koishi E 模型实测），
     全程无呼吸的图会被判"均匀体力型"。每 5 段插一段 breather，
     段内行目标由外层按 yield 折算压低。 */
  breather: {
    label: '疏流',
    init: function (s, rnd) {
      s.order = ORDERS[Object.keys(ORDERS)[randInt(3, rnd)]];
      s.k = 0;
    },
    step: function (s, i, rnd, evt) {
      var cell = evt && evt.cell != null ? evt.cell : i;
      if (cell % 2 !== 0) return [];        // 1/16 位休止
      if (cell % 4 === 2 && rnd() < 0.45) return [];  // 半数 1/8 位也休止
      return [s.order[s.k++ % 4]];
    },
    period: 4,
    yield: 0.42
  },

  /* 双押切：单键 + 双押混合，模拟 jumpstream */
  jumpstream: {
    label: '双押切',
    init: function (s, rnd) { s.k = 0; },
    step: function (s, i, rnd) {
      var b = 3 - (i % 4);
      if (i % 2 === 0) return [b];
      var partner = (b + 1 + ((i / 2) % 2 === 0 ? 0 : 2)) % 4;
      return [b, partner].sort();
    },
    period: 8
  },

    /* 跳押切：1/8 与正拍放「左右手对交替双押」（[01]↔[23]），偶尔混三押，
       非正拍走单键轮转。人类高 hs 谱的实际写法就是 [01][23][01][23] 的
       交替手对连打（每手隔行一跳，段内单手 3 次/s），偶尔 [013] 三押。
       为什么不用轮转三押：MinaCalc 按左右手（4K: 0,1 | 2,3）分别评估，
       handstream 由「单手跳押密度」驱动——人类 14–16 nps 带里单手跳押
       0.8–1.2 次/s 对应 hs/overall ≈ 0.80–0.92，0.4–0.8 只有 ≈0.68。
       轮转三押的全行视图是三押，单手视角却是「跳-单-单」碎片。 */
    handstream: {
      label: '跳押切',
      init: function (s, rnd) { s.j = 0; s.k = 0; s.kBlock = -1; },
      step: function (s, i, rnd, evt) {
        var cell = evt && evt.cell != null ? evt.cell : i;
        /* 半格单音走对侧手两列交替（与下面的单音层同源）：同手对
           （[01]/[23]）在半格上连出会形成 40ms 同列，正是 Koishi
           jack 公式最敏感的区域。 */
        if (evt && evt.sub) {
          var obS = (s.j >> 1) % 2;
          var offS = obS === 0 ? [2, 3] : [0, 1];
          return [offS[s.k++ % 2]];
        }
        if (cell % 2 === 0) {
          /* 同手连跳块：2 连同手对（如 [01][01]），再换对侧。
             MinaCalc 的 handstream 按「单手行速」计——左右每行轮换时
             单手只有 3.1 次/s；2 连块让承担手在块内跑满 6.2 次/s。
             为什么是 2 连不是 4-6 连（旧值）：站点 Koishi 实测 4 连同列
             反复 = repeat 0.616 / rotation 0.233，直接被判 jack（概率
             98.9%）；人类 tech 标杆（5898623）repeat 0.335。2 连既保
             住单手 6.2 次/s 的 hs 密度，又把同列反复压回人类带内。 */
          var block = (s.j >> 1) % 2;                    // 每 2 行换手
          var pair = block === 0 ? [0, 1] : [2, 3];
          var pos2 = s.j % 2;
          var tri = cell % 4 === 0 && pos2 === 1;        // 块中第 2 行带一个三押
          s.j++;
          if (tri) {
            var other = pair[0] < 2 ? 2 : 0;
            return pair.concat([other]).sort(function (a, b) { return a - b; });
          }
          return pair.slice();
        }
        /* 单音层：当前块的承担手的对侧两列交替——一手跳块、另一手 1/16 串。
           四列轮转会撞进双押刚用过的列，162ms 内同列就是假纵连。
           游标随块重置（旧代码 s.k 全局递增，切块后奇偶漂移，
           单手负载偏 20%——站点 E 模型实测左手 0.436 vs 右手 0.361）。 */
        var offBlk = (s.j >> 1) % 2;
        var off = offBlk === 0 ? [2, 3] : [0, 1];
        if (s.kBlock !== offBlk) { s.kBlock = offBlk; s.k = 0; }
        return [off[s.k++ % 2]];
      },
      period: 4
    },

    /* 和弦纵连：同一组双押在 1/8 拍上连续重复，是 chordjack 的唯一来源。
       真实谱的写法是「两列一组、每 1/8 拍砸一次」，中间不塞音。
       所以间隔不足 1/8 拍时返回空数组——让这一格空着，
       而不是像老版那样在 81ms 处再砸一次（那会算成 30+ 的 jackspeed）。 */
    chordjack: {
      label: '和弦纵连',
      /* 枢轴列纵连：每个和弦都含枢轴列（纵连手感的核心——同一根手指
         以 6.2 次/s 重复起压），另一列在 3 个变体间随机走。
         为什么不用「同组连砸」：整段只有 1 种排列会被 MinaCalc 的
         vibro 检测抓到（num_var=1 → 修正 ×0.5）；3 个变体只罚 ×0.95，
         同时保住 actual_jacks（相邻和弦共享枢轴列）。
         人类 D14 参照（3798610，cj MSD 30.7）就是这么写的：
         100ms 间隔、共享多数列的和弦连打。 */
      init: function (s, rnd) {
        s.pivot = randInt(4, rnd);
        s.others = [0, 1, 2, 3].filter(function (c) { return c !== s.pivot; });
        s.oi = randInt(3, rnd);
        s.lastT = -1e9;
      },
      step: function (s, i, rnd, evt) {
        var t = evt ? evt.t : i * 80;
        if (t - s.lastT < 155) return [];        // 不足 1/8 拍（162ms）就不重复
        var cell = evt && evt.cell != null ? evt.cell : 0;
        if (cell % 2 !== 0) return [];           // 1/16 位休止：双押在这里会被
                                                 // 「奇格压单键」压成同列单键，
                                                 // 连起来就是 0-0-0-2-2-2 假纵连
        s.lastT = t;
        s.oi = (s.oi + 1 + randInt(2, rnd)) % 3;
        return [s.pivot, s.others[s.oi]].sort(function (a, b) { return a - b; });
      },
      period: 2,
      yield: 0.5   // 一半的步因 155ms 间隔保护落空，配额按 2 倍补
    }
  };

  /* 键型分组：给 UI 用的分类（对应文献 2.1.2 的四大类） */
  var GROUPS = {
    '常规·叠键': ['jack', 'chordjack', 'minijack', 'longjack', 'anchor'],
    '常规·乱键': ['stream', 'splitrail'],
    '常规·切换': ['jumpstream', 'handstream', 'trill'],
    '常规·技巧': ['roll', 'ladder', 'staircase', 'jumptrill', 'splittrill']
  };

  /* ------------------------------------------------------------------ */
  /* 生成器                                                              */
  /* ------------------------------------------------------------------ */
  /**
   * @param {object} opt
   *   steps        总步数（1/16 拍）
   *   nps          目标密度 notes/second
   *   stepMs       每步毫秒
   *   weights      { patternName: weight } 键型配比
   *   sectionSteps 每段步数（默认 64 = 4 小节 @16 分）
   *   rnd          可注入随机数（便于复现）
   *   accent       Float32Array，每步的音频强度 0..1（可选）
   *   keys         4
   * @returns { columns: Array<number[]>, meta }
   */
  function generate(opt) {
    var o = opt || {};
    var steps = o.steps || 0;
    var keys = o.keys || 4;
    var stepMs = o.stepMs || 83.33;
    var rnd = o.rnd || Math.random;
    var weights = o.weights || { stream: 1 };
    var sectionSteps = o.sectionSteps || 64;
    var accent = o.accent || null;

    if (!steps) throw new Error('缺 steps');

    // 目标：每秒 nps 个音符 → 每步多少音符
    var perStep = (o.nps || 8) * (stepMs / 1000);
    var durSecTotal = (steps * stepMs) / 1000;

    /* ---- 采音约束 ----
       之前音符落在「1/16 网格的随机格子」上，听感是散的。
       现在改成：先按目标密度反推需要多少个「发声格」，再取音频强度最高的那些格。
       音高/鼓点越强的地方越可能出音符，这才是「采音」而不是「铺格子」。 */
    var keepThreshold = 0;
    if (accent && accent.length) {
      var estCells = Math.round((o.nps || 8) * durSecTotal / 1.35);  // 1.35 ≈ 平均每格音符数
      estCells = Math.max(8, Math.min(steps, estCells));
      var sortedAcc = Array.prototype.slice.call(accent).sort(function (a, b) { return b - a; });
      keepThreshold = sortedAcc[Math.min(sortedAcc.length - 1, estCells)] || 0;
      // 留一点余量，避免阈值卡太死导致密度不足
      keepThreshold *= 0.82;
    }

    // 密度曲线：每格的相对强度（1 = 平均），用来让高潮段更密、平静段更疏
    var densCurve = o.densityCurve || null;

    // 把权重展开成段落表
    var pool = [];
    Object.keys(weights).forEach(function (k) {
      var w = weights[k];
      if (!w || !PATTERNS[k]) return;
      for (var i = 0; i < Math.round(w * 10); i++) pool.push(k);
    });
    if (!pool.length) pool = ['stream'];

    var columns = new Array(steps);
    var sections = [];
    var totalNotes = 0;

    for (var s = 0; s < steps; s += sectionSteps) {
      var len = Math.min(sectionSteps, steps - s);
      var name = pool[Math.floor(rnd() * pool.length)];
      var pat = PATTERNS[name];
      var state = {};
      pat.init(state, rnd);

      var secNotes = 0;
      var skipped = 0;

      for (var i = 0; i < len; i++) {
        var idx = s + i;
        var cols;

        /* 采音：只在高强度格子上放音符 */
        var energy = accent ? Math.min(1, Math.max(0, accent[idx] || 0)) : 0.5;
        if (accent && energy < keepThreshold) { columns[idx] = []; skipped++; continue; }

        // 密度曲线调制：这一段整体偏强时，允许更密的排列
        var densMul = densCurve ? Math.max(0.35, Math.min(1.6, densCurve[idx])) : 1;

        // 弱拍上仍可能跳过（即使过了阈值），避免机械地每格都放
        var keep = Math.min(1, (perStep * densMul) / Math.max(0.5, 1 + energy * 0.55));
        if (i % 4 === 0) keep = Math.min(1, keep * 1.18);
        if (rnd() > keep) { columns[idx] = []; skipped++; continue; }

        cols = pat.step(state, i, rnd);
        // 收敛到合法列并去重
        var set = {};
        for (var k = 0; k < cols.length; k++) {
          var c = cols[k];
          if (c >= 0 && c < keys) set[c] = 1;
        }
        var arr = Object.keys(set).map(Number).sort();

        /* 密度补偿：单键型（乱键/交互/散点）每步最多 1 个音符，
           目标密度高于「每步 1 个」时永远追不上。这里按比例把一部分升级成双押。 */
        if (perStep > 1 && arr.length === 1) {
          // 高潮段（densMul 高）更容易升级成双押，平静段保持单键
          var extra = (perStep - 1) * densMul;
          if (rnd() < extra) {
            var cand = [];
            for (var e = 0; e < keys; e++) if (arr.indexOf(e) < 0) cand.push(e);
            if (cand.length) arr.push(cand[Math.floor(rnd() * cand.length)]);
            arr.sort(function (a, b) { return a - b; });
          }
        }

        columns[idx] = arr;
        totalNotes += arr.length;
        secNotes += 1;
      }

      sections.push({ name: name, label: pat.label, from: s, to: s + len, notes: secNotes, skipped: skipped });
    }

    var durSec = (steps * stepMs) / 1000;
    return {
      columns: columns,
      sections: sections,
      meta: {
        steps: steps,
        stepMs: stepMs,
        durationSec: Math.round(durSec * 10) / 10,
        notes: totalNotes,
        nps: durSec > 0 ? Math.round((totalNotes / durSec) * 100) / 100 : 0,
        targetNps: o.nps,
        alignThreshold: Math.round(keepThreshold * 1000) / 1000,
        aligned: !!accent
      }
    };
  }

  /* ------------------------------------------------------------------ *
   * generateFromPeaks —— 先定时刻、再定键位
   *
   * 为什么不用 BPM 网格：真实音乐上 BPM 检测很容易错（我实测把 145 判成 96.8，
   * 差 1.5 倍，因为附点/三连音节奏会让自相关在 1.5 倍周期上出现强峰）。
   * BPM 一旦错，网格步长就错，后面的采音全歪。
   *
   * Mug-Diffusion 干脆不用音乐网格，改用固定 46ms 时间轴 + 每列 4 通道
   * （起始位/亚帧偏移/LN体/LN尾）+ 扩散模型学「哪一帧该有音符」。
   * 这里用规则版做同样的事：
   *   1) 从细包络里找 onset 峰 → 候选时刻（精度 11.6ms，与 BPM 无关）
   *   2) 按段落能量分配配额 → 高潮段多给、平静段少给（这就是「情感调度」）
   *   3) 段内按峰值强度选点 → 最强的音一定被采到
   *   4) 在每个选中时刻上依次驱动键型步进器 → 决定按哪几列
   *
   * 产出音符的时间 = 音频峰时间，零偏移，不需要事后吸附。
   * ------------------------------------------------------------------ */
  function generateFromPeaks(opt) {
    var o = opt || {};
    var keys = Math.max(1, Math.min(18, o.keys || 4));
    var rnd = o.rnd || Math.random;
    var weights = o.weights || { stream: 1 };
    var nps = o.nps || 8;
    var fine = o.fine;
    var peaks = o.peaks || [];
    var sectionSec = o.sectionSec || 8;
    var segSec = o.segSec || 2;
    // 平均每个事件带几个音符：双押多的时候会 > 1
    var chordRate = o.chordRate != null ? o.chordRate : 1.22;

    if (!fine || !fine.energy) throw new Error('generateFromPeaks 需要 fine 包络');
    if (!peaks.length) throw new Error('没有可用的 onset 峰');

    var hop = fine.hopMs;
    var durationSec = o.durationSec || (fine.energy.length * hop / 1000);

    // ---- 事件表 ----
    var ev = [];
    var seen = {};
    /* 网格模式：事件取自信乐网格（1/16 拍），而不是任意 onset 峰。
       为什么必须先网格：用 onset 峰直接落音，声音是贴上了，但音符会落在
       23ms、46ms 这种非分音位置上，玩家的手感和读谱会散——人类谱一定是
       1/4、1/8、1/16、1/12 这些能除尽拍子的位置。
       网格模式的做法是：先按格量能量（analyzeBuffer 的 accent），
       再让段配额在格子上挑最强的那些格，键型仍在格上排。
       这样每个音符都严格落在 1/16 拍上，同时强的鼓点优先被选中。 */
    if (o.grid && o.grid.accent && o.grid.stepMs) {
      var g = o.grid;
      var gOff = g.offsetMs || 0;
      var gMin = o.gridMin != null ? o.gridMin : 0.04;
      for (var gi = 0; gi < g.accent.length; gi++) {
        var gt = gOff + gi * g.stepMs;
        if (gt > durationSec * 1000) break;
        if (gt < (o.startMs || 0)) continue;
        var ga = g.accent[gi];
        if (ga > gMin) ev.push({ t: Math.round(gt), s: ga, cell: gi });
      }
    } else {
      for (var i = 0; i < peaks.length; i++) {
        var p = peaks[i];
        if (p < 0 || p >= fine.energy.length) continue;
        var t = p * hop;
        if (t < (o.startMs || 0) || t > durationSec * 1000) continue;
        ev.push({ t: t, s: fine.energy[p] });
        seen[p] = 1;
      }
    }
    if (!ev.length) throw new Error('没有可用的事件（峰或网格都是空的）');

    var targetNotes = nps * durationSec;
    var targetEvents = Math.round(targetNotes / chordRate);
    if (targetEvents > ev.length) targetEvents = ev.length;

    // ---- 分段配额：按段均值能量加权 ----
    var nSeg = Math.max(1, Math.ceil(durationSec / segSec));
    var segMap = [];
    for (var si = 0; si < nSeg; si++) segMap.push([]);
    for (var e = 0; e < ev.length; e++) {
      var sidx = Math.min(nSeg - 1, Math.floor(ev[e].t / 1000 / segSec));
      segMap[sidx].push(e);
    }

    var segEnergy = [], totalSegE = 0;
    for (var sg = 0; sg < nSeg; sg++) {
      var lst = segMap[sg], sum = 0;
      for (var q = 0; q < lst.length; q++) sum += ev[lst[q]].s;
      segEnergy[sg] = lst.length ? sum / lst.length : 0;
      totalSegE += segEnergy[sg];
    }
    var avgSegE = totalSegE / nSeg || 1;

    // ---- 键型池 ----
    var pool = [];
    Object.keys(weights).forEach(function (k) {
      var w = weights[k];
      if (!w || !PATTERNS[k]) return;
      for (var wi = 0; wi < Math.round(w * 10); wi++) pool.push(k);
    });
    if (!pool.length) pool = ['stream'];

    var nSecAll = Math.max(1, Math.ceil(durationSec / sectionSec));
    // 每事件期望音符数：用来决定单键升级成双押/三押的概率
    var perEvent = targetEvents > 0 ? targetNotes / targetEvents : 1;
    var basePerSeg0 = targetEvents / nSeg;

    /* 段落键型预排 + 选格 + 发射 = 一个可整体重跑的单元。
       为什么要重跑：配额→选格→键型发射这条链上，chordjack 一半的步会因
       155ms 间隔保护落空（yield 0.5）、handstream 一个事件出 2 个音，
       实际音符数对键型混合极其敏感——同 nps 实测能在 12.3～14.1 之间晃。
       老代码只发射一次 + 弱补偿，密度钉不住。这里用 quotaScale 全局缩放
       迭代，取最接近目标的一次；有 rndSeed 时每轮重造同种子 RNG，
       保证 (nps, seed) 完全可复现。 */
    function runOnce(quotaScale) {
      var rr = (o.rndSeed != null) ? lcg(o.rndSeed) : rnd;
      /* 双手负载追踪器：rotstream/handstream 的手对选择与补行偏置共用。
         人类 D14 tech 图左右手 work 几乎 1:1（站点 E 模型实测 0.407/0.411、
         0.525/0.514），长期偏侧在站点诊断里是显性败项（本图旧版左手
         0.436 vs 右手 0.361，超载 20%）。每次 runOnce 重新计数。 */
      var load = { lh: 0, rh: 0 };
      /* 段落键型预排：按权重比例「洗牌发牌」，不是逐段独立抽签。
         独立抽样在段数少时方差极大（24s 段落只有 10 段，chordjack
         权重 21% 可能一段都抽不到，也可能连来四段）；人类谱师是
         按比例规划段落类型的。发牌制让段型混合精确等于权重比例，
         段落长短和混合控制就此解耦。 */
      var totalW = 0;
      Object.keys(weights).forEach(function (k) { totalW += (weights[k] || 0); });
      if (totalW <= 0) totalW = 1;
      var bag = [];
      Object.keys(weights).forEach(function (k) {
        var w = weights[k];
        if (!w || !PATTERNS[k]) return;
        var c = Math.max(0, Math.round(nSecAll * w / totalW));
        for (var bi = 0; bi < c; bi++) bag.push(k);
      });
      while (bag.length < nSecAll) bag.push(pool[Math.floor(rr() * pool.length)]);
      if (bag.length > nSecAll) bag = bag.slice(0, nSecAll);
      /* 能量-键型关联：硬键型（stream/jumpstream/handstream/rotstream，
         MSD 密度高）优先分给高能段。人类谱师把最密的段落放在歌的高潮——
         同样的谱面，硬键型落在高潮段 MSD 能差 4 分（实测 22.3 vs 18.0）。
         做法：段按能量降序走，排名越靠前抽到硬键型的概率越高
         （0.25 → 0.80 线性），两组各自洗牌后按序发。 */
      function shuffleArr(a) {
        for (var i2 = a.length - 1; i2 > 0; i2--) {
          var j2 = Math.floor(rr() * (i2 + 1));
          var t2 = a[i2]; a[i2] = a[j2]; a[j2] = t2;
        }
      }
      // 段能量代理：段内各 2s 分段的 mul 均值
      var secE = [];
      for (var se = 0; se < nSecAll; se++) {
        var lo0 = Math.floor(se * sectionSec / segSec), hi0 = Math.min(nSeg - 1, Math.floor((se + 1) * sectionSec / segSec) - 1);
        var sE = 0, cE = 0;
        for (var dE = lo0; dE <= hi0; dE++) if (segMap[dE] && segMap[dE].length) {
          var rE = Math.max(0.02, segEnergy[dE] / avgSegE);
          sE += Math.max(0.01, Math.min(2.8, Math.pow(rE, o.dynExp || 1))); cE++;
        }
        secE.push(cE ? sE / cE : 0);
      }
      var order = secE.map(function (v, i2) { return i2; })
                     .sort(function (a, b) { return secE[b] - secE[a]; });
      /* 呼吸段注入：约每 5 段把一段换成 breather（疏流），优先换低能段——
         人类 D14 tech 图的 recovery_ge0.5 ≈ 0.02-0.03（站点 Koishi E 模型
         实测 5898623 = 0.031），全程 1/16 密铺的图会被判"均匀体力型"。
         呼吸段放在能量低谷处——谱师"高潮前收一拍"的写法。 */
      if (o.breath !== false && nSecAll >= 6) {
        var nBreath = Math.max(1, Math.round(nSecAll / 5));
        var lowFirst = secE.map(function (v, i2) { return i2; })
                          .sort(function (a, b) { return secE[a] - secE[b]; });
        for (var bi3 = 0; bi3 < nBreath && bi3 < lowFirst.length; bi3++) {
          var bi4 = lowFirst[bi3];
          if (bag[bi4] !== 'breather') bag[bi4] = 'breather';
        }
      }
      /* 硬/软分组放在呼吸段注入之后：breather 是软键型，不能进 HARD 组
         和 stream 抢高能段；rotstream 是硬键型（1/16 双押流，密度最高）。 */
      var HARD = { stream: 1, jumpstream: 1, handstream: 1, rotstream: 1 };
      var bagH = [], bagS = [];
      for (var bi2 = 0; bi2 < bag.length; bi2++) (HARD[bag[bi2]] ? bagH : bagS).push(bag[bi2]);
      shuffleArr(bagH); shuffleArr(bagS);
      var assign = new Array(nSecAll);
      var pH = 0, pS = 0;
      for (var rk = 0; rk < order.length; rk++) {
        var z = order.length > 1 ? 1 - rk / (order.length - 1) : 0.5;
        var pHard = 0.25 + 0.55 * z;
        var nm;
        if (rr() < pHard && pH < bagH.length) nm = bagH[pH++];
        else if (pS < bagS.length) nm = bagS[pS++];
        else if (pH < bagH.length) nm = bagH[pH++];
        else nm = pool[Math.floor(rr() * pool.length)];
        assign[order[rk]] = nm;
      }
      var secPats = [];
      for (var spi = 0; spi < nSecAll; spi++) {
        secPats.push({ name: assign[spi], yield: Math.max(0.2, PATTERNS[assign[spi]].yield || 1) });
      }
      function patOf(tSec) {
        var si = Math.min(nSecAll - 1, Math.max(0, Math.floor(tSec / sectionSec)));
        return secPats[si];
      }

      /* 每段的行数目标（能量加权 × 出勤率）：
         补行补偿的分母。chordjack 段按 0.5 折算——纵连受 1/8 拍物理上限
         （6.16 行/s），硬填满会把「和弦-单键-和弦」拆成散点，反而不像人。
         纵连段少掉的行数由标定循环全局补（quotaScale 上调），
         也就是让别的段密一点，而不是在纵连间隙里塞单键。 */
      var segMul = [];
      for (var d2 = 0; d2 < nSeg; d2++) {
        var r2 = Math.max(0.02, segEnergy[d2] / avgSegE);
        var m2 = Math.pow(r2, o.dynExp || 1);
        segMul[d2] = Math.max(0.01, Math.min(o.segMulCap || 2.8, m2));
      }
      var secRowTarget = [];
      var secMul = [];           // 段能量乘数（行加宽档位用）
      for (var s2 = 0; s2 < nSecAll; s2++) {
        var lo = Math.floor(s2 * sectionSec / segSec), hi = Math.min(nSeg - 1, Math.floor((s2 + 1) * sectionSec / segSec) - 1);
        var sum = 0, wsum = 0, wn = 0;
        for (var d3 = lo; d3 <= hi; d3++) if (segMap[d3] && segMap[d3].length) {
          sum += basePerSeg0 * segMul[d3];
          wsum += segMul[d3]; wn++;
        }
        secRowTarget[s2] = sum * secPats[s2].yield;
        secMul[s2] = wn ? wsum / wn : 1;
      }

      /* 按音频写（听写式选格）：逐拍决定细分层级，不再按配额散挑。
         人类谱师的写法是「听写」——正拍一定踩，音乐确实演奏了 1/8 就跟
         1/8，爆发段把 1/16 整串收进来，音乐休止整拍留白。节奏与音乐
         连续对应，而不是「能量强的格子东一个西一个」的切分拼贴。
         旧版按配额散挑，韵律点（连续的踩镲/旋律线）会被拆散，
         听感上就不是这首歌的节奏。
         密度旋钮（quotaScale）映射成阈值松紧：k8/k16 越低，
         越弱的 1/8、1/16 事件也被跟进——单调，可被外层标定循环驱动。 */
      var picked = [];
      /* subKnob 允许负值：1/32 半格事件按段能量额外加密度（见下方
         pstream 构建），标定循环需要能反向压低主网格选格数，
         否则总音符数只涨不落，nps 目标永远收敛不到。 */
      var subKnob = Math.max(-1.2, Math.min(2.0, quotaScale - 0.5));
      for (var d = 0; d < nSeg; d++) {
        if (!segMap[d].length) continue;
        var mulD = Math.max(0.01, Math.min(2.8, segMul[d]));
        var k8 = Math.max(0.08, 0.60 - 0.28 * subKnob - 0.06 * (mulD - 1));
        var k16 = Math.max(0.0, 0.72 - 0.38 * subKnob - 0.10 * (mulD - 1));
        var byBeat = {};
        for (var si = 0; si < segMap[d].length; si++) {
          var idxC = segMap[d][si];
          var cc = ev[idxC].cell;
          var bb = cc >> 2, pp = cc & 3;
          if (!byBeat[bb]) byBeat[bb] = [null, null, null, null];
          if (byBeat[bb][pp] == null || ev[idxC].s > ev[byBeat[bb][pp]].s) byBeat[bb][pp] = idxC;
        }
        var beatKeys = Object.keys(byBeat).map(Number).sort(function (a, b) { return a - b; });
        for (var bk = 0; bk < beatKeys.length; bk++) {
          var q = byBeat[beatKeys[bk]];
          if (q[0] != null) picked.push(q[0]);           // 正拍一定踩
          var e0 = q[0] != null ? ev[q[0]].s : 0;
          var e2 = q[2] != null ? ev[q[2]].s : 0;
          if (q[2] != null && e2 >= k8 * Math.max(e0, 0.04)) picked.push(q[2]);
          var eMx = Math.max(e0, e2, 0.04);
          for (var sp = 1; sp <= 3; sp += 2) {           // 1/16 位：1、3
            if (q[sp] != null && ev[q[sp]].s >= k16 * eMx) picked.push(q[sp]);
          }
        }
      }
      picked.sort(function (a, b) { return ev[a].t - ev[b].t; });

      /* ---- 1/32 半格 + 1/48 三分格细分（爆发段密度）----
         站点 Koishi 的 D 值主要由行频驱动。v15 的结构已翻盘成 speed
         （alternationRatio 0.52 / chord1Ratio 0.73，贴近人类 tech 标杆
         5898623 的 0.49 / 0.67），但行频锁在纯 1/16 格：
         rowHz_mean 12.2（人类 19.2）、p100 13.5（人类 38.5），
         D 直接掉出可测区间（BelowScope）。
         人类 D14 tech 图的爆发段实证有更细分音——5898623（178.5bpm）
         gap=42ms（1/32）占 20%、最热处掉到 26-28ms（≈1/48）。
         所以爆发段在已选格内补单音事件：强度取 0.7-0.8×（不参与
         选格配额，只加密度），键型层强制单音。半格与三分格互斥，
         概率随段能量爬升，低能段保持纯 1/16（呼吸感由 breather
         + 选格听写共同保证）。 */
      var pstream = [];
      for (var ps2 = 0; ps2 < picked.length; ps2++) pstream.push(ev[picked[ps2]]);
      if (o.grid && g) {
        for (var ph = 0; ph < picked.length; ph++) {
          var eP = ev[picked[ph]];
          if (eP.cell == null) continue;
          var sgIdx = Math.min(nSeg - 1, Math.max(0, Math.floor(eP.t / 1000 / segSec)));
          var sd2 = Math.max(0.01, Math.min(2.8, segMul[sgIdx]));
          /* 四分音 palette（每格最多补一个，互斥）：
             人类 D14 tech 标杆 5898623 的 gap 直方图（178.5bpm，
             1/16=84ms / 1/32=42ms / 1/24=56ms / 1/48=28ms）：
               42ms 31.4%  84ms 29.1%  56ms 15.6%  33ms 9.3%
               27-31ms 2.0%  168ms 5.2%
             也就是 1/16 主网格上叠 1/32 直竖、1/24 三连摇摆、1/48 急奏
             四种细分——rhythmEntropy 2.47 就是这么来的（只有 1/16+1/32
             两种的图熵值 1.6，听着像电子钟）。
             +2/3 位（54ms）是关键：相邻两格都取 +2/3 就是 54/27ms 交替的
             三连摇摆，和 +1/2 直竖、+1/3 急奏混出完整的节奏词汇表。
             所有细分与主网格、彼此间距都 ≥27ms，不会出现乱码。 */
          var r3 = rr(), subT = -1, subS = 0;
          if (sd2 > 2.2) {
            /* 最热段：85% 的格带细分，1/24 摇摆为主（40%）、
               1/32 直竖（35%）、1/48 急奏（10%）——人类 5898623
               爆发段的配比（31/16/11）的 185bpm 对应版。 */
            if (r3 < 0.10) { subT = 1 / 3; subS = 0.7; }
            else if (r3 < 0.50) { subT = 2 / 3; subS = 0.7; }
            else if (r3 < 0.85) { subT = 0.5; subS = 0.8; }
          } else if (sd2 > 1.6) {
            if (r3 < 0.06) { subT = 1 / 3; subS = 0.7; }
            else if (r3 < 0.40) { subT = 2 / 3; subS = 0.7; }
            else if (r3 < Math.min(0.95, 0.40 + 0.55 * halfScale)) { subT = 0.5; subS = 0.8; }
          } else if (sd2 > 1.3) {
            if (r3 < 0.25) { subT = 2 / 3; subS = 0.7; }
            else if (r3 < 0.25 + 0.30 * halfScale) { subT = 0.5; subS = 0.8; }
          } else if (r3 < 0.12) {
            subT = 0.5; subS = 0.8;
          }
          if (subT >= 0) {
            var subCode = subT === 0.5 ? 1 : (subT === 2 / 3 ? 3 : 2);
            pstream.push({ t: Math.round(gOff + (eP.cell + subT) * g.stepMs),
                           s: eP.s * subS, cell: eP.cell, sub: subCode });
          }
        }
        pstream.sort(function (a, b) { return a.t - b.t; });
      }

      // ---- 逐事件驱动键型 ----
      var notesOut = [];
      var sections = [];
      var pat = null, state = null, secName = '', secStart = -1e9, secNotes = 0, secStep = 0;
      var totalNotes = 0;
      var rowsOut = 0;        // 已发射的行数（补行补偿的分母）
      var gridRows = 0;       // 主网格行数（不含半格，和弦率的分母）
      var gridNotes = 0;      // 主网格音符数（不含半格，和弦率的分子）
      var lastCol = -1;       // 上一个音符的列（插行时避开，防假纵连）
      var prevCol = -1;       // 上上行（补行避开两列）
      var usedTimes = {};     // 已占用时刻（插行不和别人撞行）
      var recentCols = [];    // 最近升级用过的列（和弦排列轮换，防 pattern_chord 虚高）
      var secStartRows = 0;   // 当前段开始时的行数（段内行进度的基准）
      var secStartT = 0;      // 当前段第一个事件的时刻
      var secIdx = 0;         // 当前段的索引（段内锁定，中途不跳）

      for (var pi = 0; pi < pstream.length; pi++) {
        var evt = pstream[pi];
        var tSec = evt.t / 1000;

        if (!pat || tSec - secStart >= sectionSec) {
          if (pat) sections.push({ label: secName, start: Math.round(secStart * 1000), notes: secNotes });
          secStart = tSec;
          secStartT = tSec;
          secStartRows = rowsOut;
          secIdx = Math.min(nSecAll - 1, Math.max(0, Math.floor(tSec / sectionSec)));
          secNotes = 0;
          secStep = 0;
          var sp = patOf(tSec);
          secName = sp.name;
          pat = PATTERNS[secName];
          state = {};
          /* ctx 透传：共享负载追踪器 + 段级参数。rotstream 的手对偏置和
             三押/双押比例从这里读——段能量高时三押点缀密（爆发段敢排），
             低能段退回双押为主。 */
          pat.init(state, rr, {
            load: load,
            secMul: secMul[secIdx] || 1,
            triRate: o.triRate != null ? o.triRate : 0.22,
            dblRate: o.dblRate != null ? o.dblRate : 0.72
          });
        }

        var cols = pat.step(state, secStep++, rr, evt) || [];
        var set = {};
        for (var ci = 0; ci < cols.length; ci++) {
          var c = cols[ci];
          if (c >= 0 && c < keys) set[c] = 1;
        }
        var arr = Object.keys(set).map(Number).sort(function (a, b) { return a - b; });
        /* chordjack 的 155ms 间隔保护会返回空（1/8 拍内不重复砸）——
           这一格就是休止，人类谱的纵连间隙也是这样。密度损失不在这里补，
           而是在配额里按 yield 放大该段的选格数（见 secPats）。 */
        if (!arr.length) continue;

        /* 1/16 网格上相邻格只隔 81ms（185 BPM）。旧规则「奇格一律压单键」
           把 chordRate 锁死 1.40（人类 D14 tech 图 2.00）——人类谱的
           1/16 位双押是常态（5898623 的 c2=0.45 约一半在奇格）。
           现在只禁三押以上：奇格留双押（81ms 双押连击，jumpstream
           手感），三押留给 1/8 以上位置。 */
        var onBeatHalf = !o.grid || evt.cell == null || (evt.cell % 2 === 0);
        if (!onBeatHalf && arr.length > 2) arr = arr.slice(0, 2);

        /* 格内细分事件（1/32 半格 sub=1 / 1/48 三分格 sub=2）强制单音：
           人类 D14 tech 图的细分串是单音流主体（5898623 纯 tap），
           双押/三押留给 1/16 主网格。
           另加同列保险：Koishi 的 jack 公式对 dtSame≈27-40ms 极敏感
           （190/(27+35)≈3.1^n），梯子型键型的游标在相邻步可能同列
           （bounce 序列 3,3），轮转型虽不会，仍统一检查最近两行。 */
        if (evt.sub && arr.length > 1) arr = arr.slice(0, 1);
        if (evt.sub && arr.length === 1 &&
            (arr[0] === lastCol || arr[0] === prevCol)) {
          arr[0] = (arr[0] + 1) % keys;
          if (arr[0] === lastCol || arr[0] === prevCol) arr[0] = (arr[0] + 2) % keys;
        }
        /* 单音禁同列连击（站点 Koishi 的 repeats 口径：紧邻上一行同列
           即记一次 jack）。人类 D14 tech 只有 0.8%，v18 有 5.0%——
           来源是梯子游标的 bounce（3,3）、钩子的循环回环、补行随机列
           撞上下一行。轮转型键型（rotstream/stream/roll）本就不会同列，
           这条规则对它们零影响；双押/三押不禁——chordjack 的枢轴纵连
           就是要靠同列反复，禁了就没这个键型了。 */
        if (arr.length === 1 && arr[0] === lastCol) {
          arr[0] = (arr[0] + 1) % keys;
          if (arr[0] === prevCol) arr[0] = (arr[0] + 1) % keys;
        }

        /* 密度补偿：按「进度」补行。行数目标按段内能量进度算
           （secRowTarget），不按全局进度——高能段该密、真空段该稀，
           全局进度会把两者抹平。 */
        var secProg = Math.max(0, Math.min(1, (tSec - secStartT) / sectionSec));
        var expectedRows = secStartRows + secProg * secRowTarget[secIdx];
        if (!evt.sub && rowsOut < expectedRows - 0.5 &&
            o.grid && evt.cell != null && g && pi + 1 < pstream.length) {
          /* 补行格在 (本格, 下一个 picked 格) 的开区间里找第一个没用过的——
             旧代码只查 cell+1，而 picked 事件密集时间隔就是 1 格，
             补行永远撞上下一事件 → 补行形同虚设，密度全压在加宽上，
             chordSize 被推到 1.78（人类 tech 1.40）。 */
          var nxt = pstream[pi + 1];
          var nextCell = nxt.cell != null ? nxt.cell : null;
          var it = -1;
          if (nextCell != null && nextCell > evt.cell + 1) {
            for (var gc = evt.cell + 1; gc < nextCell; gc++) {
              var gt2 = Math.round(gOff + gc * g.stepMs);
              if (!usedTimes[gt2]) { it = gt2; break; }
            }
          }
          if (it < 0 && (nextCell == null || nextCell > evt.cell + 1)) {
            var gt3 = Math.round(gOff + (evt.cell + 1) * g.stepMs);
            if (!usedTimes[gt3]) it = gt3;
          }
          if (it >= 0 && it < durationSec * 1000) {
            /* 避开前两行的列：补行常落在纵连间隙里，只避 lastCol 时
               会与前前行同列连击（162ms 内三连单键同列 = 假纵连）。 */
            var ic = randInt(keys, rr);
            if (ic === lastCol || ic === prevCol) {
              ic = (ic + 1) % keys;
              if (ic === lastCol || ic === prevCol) ic = (ic + 1) % keys;
            }
            /* 负载偏置：补列若全落在重载侧，翻到对侧列。
               人类 D14 tech 图左右手 work 差 <5%（0.407/0.411）。 */
            if (load.lh - load.rh > 0.6 && ic < 2) ic += 2;
            else if (load.rh - load.lh > 0.6 && ic >= 2) ic -= 2;
            notesOut.push({ t: it, col: ic });
            usedTimes[it] = 1;
            totalNotes++; rowsOut++; secNotes++;
            if (ic < 2) load.lh++; else load.rh++;
            prevCol = lastCol;
            lastCol = ic;
          }
        }
        /* 行补偿与行加宽必须独立判断。
           加宽的目标函数从「总音符数救急」换成「网格行和弦率达标」：
           半格细分每加一个单音就把 totalNotes 推高 1，但它不承担和弦率
           ——旧的 totalNotes<expected-8 条件被半格「喂饱」后永不触发，
           chordRate 掉到 1.08（人类 tech 1.40），站点 Koishi 的
           chord1Ratio 冲到 0.85（人类 0.67）。
           现在直接盯网格行：gridNotes 落后 gridRows×chordRate 就把当前
           行加宽。仍只在中高能段发动（mul>1.15）、仍限 2-3 键
           （人类 D14 tech：单音 67% / 双押 26% / 三押 7%）。 */
        if (!evt.sub && arr.length < keys && onBeatHalf &&
            gridNotes < gridRows * chordRate - 0.5) {
          var mulNow = secMul[secIdx] || 1;
          if (mulNow > 1.15) {
            /* 三押门槛从 2.2 降到 1.8：人类 D14 tech 的 chord3Ratio
               是 0.067（5898623），v17 只有 0.013——三押几乎绝迹，
               爆发段的和弦色彩出不来。1.8 以上的段才允许三押，
               仍受 maxW=3 和 keys=4 的物理限制。 */
            var maxW = mulNow > 1.8 ? 3 : 2;
            var upg = 0;
            while (arr.length < maxW && arr.length < keys &&
                   (upg === 0 || (gridNotes < gridRows * chordRate - 3 && rr() < 0.7))) {
              var cand = [];
              for (var ce = 0; ce < keys; ce++) if (arr.indexOf(ce) < 0) cand.push(ce);
              if (!cand.length) break;
              /* 优先取最近两行没用过的列；全用过才退回随机 */
              var fresh = cand.filter(function (c2) {
                return recentCols.indexOf(c2) < 0;
              });
              var pickFrom = fresh.length ? fresh : cand;
              /* 负载偏置：加宽列优先补到轻载手——人类 D14 tech 左右手
                 work 差 <5%，旧版补列纯随机，实测偏侧 13%。 */
              if (pickFrom.length > 1 && Math.abs(load.lh - load.rh) > 0.4) {
                var want = load.lh - load.rh > 0.4 ? 'r' : 'l';
                var side = pickFrom.filter(function (c2) {
                  return want === 'r' ? c2 >= 2 : c2 < 2;
                });
                if (side.length) pickFrom = side;
              }
              arr.push(pickFrom[Math.floor(rr() * pickFrom.length)]);
              upg++;
              if (upg >= 2) break;
            }
            if (upg) {
              arr.sort(function (a, b) { return a - b; });
              for (var rc3 = 0; rc3 < arr.length; rc3++) recentCols.push(arr[rc3]);
              if (recentCols.length > 8) recentCols.splice(0, recentCols.length - 8);
            }
          }
        }

        var tt = Math.round(evt.t);
        for (var nk = 0; nk < arr.length; nk++) {
          notesOut.push({ t: tt, col: arr[nk] });
          if (arr[nk] < 2) load.lh++; else load.rh++;
        }
        usedTimes[tt] = 1;
        rowsOut++;
        if (!evt.sub) { gridRows++; gridNotes += arr.length; }
        prevCol = lastCol;
        lastCol = arr[arr.length - 1];
        totalNotes += arr.length;
        secNotes += arr.length;
      }
      if (pat) sections.push({ label: secName, start: Math.round(secStart * 1000), notes: secNotes });
      notesOut.sort(function (a, b) { return a.t - b.t || a.col - b.col; });
      return { notes: notesOut, sections: sections, totalNotes: totalNotes,
               rows: rowsOut, picked: picked.length, stream: pstream.length,
               handLoad: { lh: load.lh, rh: load.rh } };
    }

    var best = null, halfScale = 1.0;
    /* 标定尝试 10 次：两个旋钮（选格阈值 + 半格概率）都是非线性的，
       8 次在密度目标跨度大时收敛不到 1.5% 内。
       每次重放同种子随机数（rndSeed），(nps, seed) 仍完全可复现。 */
    var attempts = o.rndSeed != null ? 10 : 1;
    /* 目标密度超过纯网格物理容量时（候选格 × 和弦率 1.42 ≈ 16.3 nps
       上限），选格阈值直接饱和到 3.0——候选格全收，标定预算全留给
       半格概率。阈值旋钮再拧也没有更多格子，不饱和的话 10 次尝试
       全耗在爬阈值上，半格概率一次都调不到。 */
    var quotaScale = targetNotes > ev.length * (chordRate + 0.15) ? 3.0 : 1.0;
    for (var att = 0; att < attempts; att++) {
      var res = runOnce(quotaScale);
      if (!best || Math.abs(res.totalNotes - targetNotes) < Math.abs(best.totalNotes - targetNotes)) {
        best = res;
      }
      var err = res.totalNotes / Math.max(1, targetNotes);
      if (Math.abs(err - 1) < 0.015) break;
      /* 收敛步长减半防过冲：单音流键型的产出率对阈值非线性（k8/k16 降一点，
         选格数跳一截），大步长会在 1.6↔0.7 之间振荡。 */
      var step = err > 1 ? 1 / (1 + (err - 1) * 0.5) : 1 / (1 - (1 - err) * 0.5);
      var qs2 = Math.max(0.25, Math.min(3.0, quotaScale * step));
      /* 第二标定维度：半格细分概率。欠密度且阈值已饱和（不再增长）
         时，唯一还能加密度的就是 1/32 半格概率——否则高 nps 目标
         永远收敛不到，D 值卡在 BelowScope。过密度时同步回收。 */
      if (err < 0.985 && qs2 >= quotaScale - 1e-9) {
        halfScale = Math.min(1.8, halfScale * 1.25);
      } else if (err > 1.02) {
        halfScale = Math.max(0.5, halfScale * 0.94);
      }
      quotaScale = qs2;
    }
    var notesOut = best.notes;
    var sections = best.sections;

    // 音符所在帧的平均能量 ÷ 全局均值，直接作为采音对齐的预期值
    var sumAll = 0;
    for (var q2 = 0; q2 < fine.energy.length; q2++) sumAll += fine.energy[q2];
    var avgAll = sumAll / fine.energy.length || 1e-9;
    var sumNote = 0, cntNote = 0;
    for (var q3 = 0; q3 < notesOut.length; q3++) {
      var idx2 = Math.round(notesOut[q3].t / hop);
      if (idx2 >= 0 && idx2 < fine.energy.length) { sumNote += fine.energy[idx2]; cntNote++; }
    }
    var alignRatio = cntNote ? (sumNote / cntNote) / avgAll : 0;

    /* 单手体检：MinaCalc 按左右手分开评估（4K: 列 0,1 = 左手，2,3 = 右手），
       handstream 由「单手跳押密度」驱动（人类 14–16 nps 带：单手 0.8–1.2
       次/s ↔ hs/overall ≈ 0.80–0.92）。jps 双侧不对称说明图案偏侧，
       谱师会把它当败笔。 */
    var masks = {};
    for (var h1 = 0; h1 < notesOut.length; h1++) {
      var nt = notesOut[h1];
      masks[nt.t] = (masks[nt.t] || 0) | (1 << nt.col);
    }
    var lhTaps = 0, rhTaps = 0, lhJump = 0, rhJump = 0;
    for (var h2 in masks) {
      var mk = masks[h2];
      var lc = mk & 3, rc2 = (mk >> 2) & 3;
      lhTaps += (lc & 1) + ((lc >> 1) & 1);
      rhTaps += (rc2 & 1) + ((rc2 >> 1) & 1);
      if (lc === 3) lhJump++;
      if (rc2 === 3) rhJump++;
    }
    var hand = {
      lhNps: durationSec > 0 ? Math.round(lhTaps / durationSec * 100) / 100 : 0,
      rhNps: durationSec > 0 ? Math.round(rhTaps / durationSec * 100) / 100 : 0,
      lhJps: durationSec > 0 ? Math.round(lhJump / durationSec * 100) / 100 : 0,
      rhJps: durationSec > 0 ? Math.round(rhJump / durationSec * 100) / 100 : 0,
      jps: durationSec > 0 ? Math.round(((lhJump + rhJump) / 2 / durationSec) * 100) / 100 : 0
    };

    return {
      notes: notesOut,
      sections: sections,
      meta: {
        mode: 'onset',
        durationSec: Math.round(durationSec * 10) / 10,
        notes: notesOut.length,
        nps: durationSec > 0 ? Math.round((notesOut.length / durationSec) * 100) / 100 : 0,
        targetNps: nps,
        events: best.stream,
        candidates: ev.length,
        rows: best.rows,
        rowsPerSec: durationSec > 0 ? Math.round((best.rows / durationSec) * 100) / 100 : 0,
        perEvent: Math.round(perEvent * 100) / 100,
        alignRatio: Math.round(alignRatio * 100) / 100,
        hopMs: hop,
        hand: hand,
        /* 左右手音符数比：人类 D14 tech 图两手 work 差 <5%（站点 E 模型
           0.407/0.411、0.525/0.514）。>1.15 视为偏侧败笔。 */
        handRatio: (best.handLoad && best.handLoad.rh > 0)
          ? Math.round(best.handLoad.lh / best.handLoad.rh * 1000) / 1000 : null
      }
    };
  }

  /* 从「每步列数组」还原成 notes（给 Scene3D / .osu 导出用）
   *
   * opt.fine    = { energy, hopMs }  高分辨率包络，启用「时间吸附」
   * opt.peaks   = [idx...]           细包络上的峰值帧（可选，优先吸附到这些点）
   * opt.snapMs  = 吸附窗口半径（默认 stepMs*0.35，上限 45ms）
   *
   * 为什么要吸附：音乐网格（1/16 拍 @ 180bpm = 83ms）比鼓点间隔粗，
   * 音符落在格子上时离真实重音可能差 30–40ms，听感就是「没踩上」。
   * Mug-Diffusion 用每列 4 通道里的 sub-frame offset 解决同一个问题；
   * 这里用「吸附到最近的音频峰」等价地解决，且不增加模型负担。
   */
  function toNotes(columns, stepMs, offsetMs, opt) {
    var o = opt || {};
    var off = offsetMs || 0;
    var fine = o.fine && o.fine.energy && o.fine.hopMs ? o.fine : null;
    var peaks = o.peaks || null;
    var snap = o.snapMs != null ? o.snapMs : Math.min(stepMs * 0.35, 45);
    var minGap = o.minGapMs != null ? o.minGapMs : 18;
    var out = [];
    var lastT = -1e9;
    var snapped = 0, total = 0;

    for (var i = 0; i < columns.length; i++) {
      var cols = columns[i];
      if (!cols || !cols.length) continue;

      var base = off + i * stepMs;
      var t = base;

      if (fine) {
        var hop = fine.hopMs;
        var lo = Math.max(0, Math.floor((base - snap) / hop));
        var hi = Math.min(fine.energy.length - 1, Math.ceil((base + snap) / hop));
        var bestIdx = -1;

        // 优先：窗口内最近的 onset 峰
        if (peaks && peaks.length) {
          var bestD = 1e9;
          for (var pi = 0; pi < peaks.length; pi++) {
            var p = peaks[pi];
            if (p < lo) continue;
            if (p > hi) break;
            var d2 = Math.abs(p * hop - base);
            if (d2 < bestD) { bestD = d2; bestIdx = p; }
          }
        }
        // 退路：窗口内能量最高的帧
        if (bestIdx < 0 && hi >= lo) {
          var bestE = -1;
          for (var f = lo; f <= hi; f++) {
            if (fine.energy[f] > bestE) { bestE = fine.energy[f]; bestIdx = f; }
          }
        }
        if (bestIdx >= 0) {
          var cand = bestIdx * hop;
          // 防重叠：太近就保留原来的格点
          if (cand - lastT >= minGap) { t = cand; snapped++; }
        }
      }

      total++;
      lastT = t;
      for (var k = 0; k < cols.length; k++) {
        out.push({ t: Math.round(t), col: cols[k] });
      }
    }

    // 附带统计，方便对比吸附前后的采音对齐度
    out.snapStats = { steps: total, snapped: snapped, snapMs: snap };
    return out;
  }

  return {
    PATTERNS: PATTERNS,
    GROUPS: GROUPS,
    ORDERS: ORDERS,
    generate: generate,
    generateFromPeaks: generateFromPeaks,
    toNotes: toNotes,
    version: '1.0.0'
  };
});
