// v2: mentor's structure. No title slide. 1 motivation slide, 5 trick slides (one per trick), then results.
const pptxgen = require('pptxgenjs');
const L = require('./lib');
const { A, C, FONT, HFONT, base, title, timeTag, caption, body, card, badge, clipPlaceholder, notes, stat } = L;

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9'; // 10 x 5.625 in
pres.title = 'Bag of Tricks for Training-Free VLA Models, accompanying video (v2)';

const SUB = !!process.env.SUBS;
const WALL = !!process.env.WALL;
const W = WALL ? require('./walls') : null;
const SECS = [];
let N = 0;
const S = (secs, dark = false) => { const s = pres.addSlide(); N += 1; base(s, dark); SECS.push(secs); return s; };
const chip = (s, text, x, y, w, kind = 'neutral', h = 0.42, size = 13) => {
  const map = { pos: [C.greenSoft, C.green], neg: [C.redSoft, C.red], neutral: [C.card, C.ink], teal: [C.tealSoft, C.teal], warn: [C.orangeSoft, C.orange], blue: [C.blueSoft, C.blue], dark: [C.dark, C.darkText] };
  const [fill, col] = map[kind];
  s.addShape('roundRect', { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.1 });
  s.addText(text, { x, y, w, h, fontFace: FONT, fontSize: size, bold: true, color: col, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
};
const axisTag = (s, text, kind) => chip(s, text, 6.9, 0.42, 2.6, kind, 0.4, 12);
const qTag = (s, text, x, y) => chip(s, text, x, y, 1.35, 'dark', 0.4, 11);
// Timeline of policy calls (filled = call, hollow = held)
const timeline = (s, dots, x, y, colorFilled, gates) => {
  dots.forEach((d, i) => {
    const xx = x + i * 0.3;
    s.addShape('ellipse', { x: xx, y, w: 0.24, h: 0.24, fill: { color: d ? colorFilled : C.white }, line: { color: d ? colorFilled : '9AA3AE', width: 1.25 } });
    if (gates && !d) s.addShape('ellipse', { x: xx + 0.07, y: y + 0.32, w: 0.1, h: 0.1, fill: { color: C.green }, line: { color: C.green, width: 0 } });
  });
};
// Native legend entries: [label, hex]
const legend = (s, entries, x, y, colW, size = 10) => {
  entries.forEach(([lab, col], i) => {
    const xx = x + i * colW;
    s.addShape('rect', { x: xx, y: y + 0.05, w: 0.16, h: 0.16, fill: { color: col }, line: { color: col, width: 0 } });
    s.addText(lab, { x: xx + 0.22, y, w: colW - 0.25, h: 0.26, fontFace: FONT, fontSize: size, color: C.ink, margin: 0, isTextBox: true, valign: 'middle' });
  });
};
const TRICK_COLORS = [['Original', '222222'], ['Foveation', '0072B2'], ['Action repeat', 'D55E00'], ['Depth pruning', '009E73'], ['Guarded reuse', 'CC79A7'], ['Temporal fusion', '8C650F']];

// ═════════════════════════ 1. MOTIVATION (15 s)
{
  const s = S(WALL ? 13 : 15);
  caption(s, 'Bag of Tricks for Training-Free Vision-Language-Action Models', 0.5, 0.12, 7.8, { size: 10 });
  s.addText([{ text: 'Motivation', options: { bold: true } }, { text: '  Efficient VLA inference without retraining', options: { bold: false } }], { x: 0.5, y: 0.32, w: 9.0, h: 0.7, fontFace: HFONT, fontSize: 26, color: C.ink, margin: 0, isTextBox: true, valign: 'middle' });
  s.addImage({ path: A('m_pipeline.png'), x: 0.5, y: 1.1, w: 9.0, h: 0.47 });
  caption(s, 'The control loop of a frozen VLA policy (Fig. 2 of the paper). Green, orange, purple, and blue boxes mark where the five tricks enter; Redundancy Control is action repeat.', 0.5, 1.7, 9.0, { size: 11, h: 0.4 });
  const red = 'C00000';
  s.addText([{ text: '[1] ', options: { bold: true, color: red } }, { text: 'VLA inference is slow. ', options: { bold: true, color: red } }, { text: 'Tens to hundreds of milliseconds per step (60 to 450 ms in our runs), which limits how often the policy can observe, decide, and correct.', options: { color: C.ink } }],
    { x: 0.5, y: 2.45, w: 9.0, h: 0.5, fontFace: FONT, fontSize: 14, margin: 0, isTextBox: true, valign: 'middle' });
  s.addText([{ text: '[2] ', options: { bold: true, color: red } }, { text: 'Gains are reported in isolation. ', options: { bold: true, color: red } }, { text: 'Training-free tricks are typically evaluated for a specific trick, backbone, benchmark, and implementation, so transfer across models is unknown.', options: { color: C.ink } }],
    { x: 0.5, y: 3.1, w: 9.0, h: 0.6, fontFace: FONT, fontSize: 14, margin: 0, isTextBox: true, valign: 'middle' });
  s.addShape('roundRect', { x: 0.5, y: SUB ? 4.05 : 4.15, w: 9.0, h: SUB ? 0.9 : 0.95, fill: { color: 'FFF0F7' }, line: { color: 'E0369A', width: 1.5 }, rectRadius: 0.1 });
  s.addText([{ text: 'Do the gains transfer across VLA models and environments, ', options: { color: C.ink } }, { text: 'or are they configuration-dependent?', options: { bold: true, color: 'E0369A' } }],
    { x: 0.7, y: SUB ? 4.05 : 4.15, w: 8.6, h: SUB ? 0.9 : 0.95, fontFace: FONT, fontSize: 16, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  notes(s, N, WALL ? 13 : 15,
    'Vision-language-action models are capable but slow. Every policy call runs a vision encoder and a language backbone. Training-free tricks promise cheaper inference, but gains are typically reported for one backbone and one benchmark. Do they transfer?',
    'Static. Optional: play a short original-policy rollout behind the loop diagram. Pipeline row cropped from Fig. 2(a).');
}

// ═════════════════════════ 2. FOVEATION (11 s)
{
  const s = S(11);
  title(s, 'Trick 1  Visual foveation');
  axisTag(s, 'What to see', 'blue');
  s.addImage({ path: A('m_fov.png'), x: 0.5, y: 1.05, w: 4.3, h: 2.72 });
  body(s, ['Sharp central disc, progressively blurred periphery', 'Keep ratio 20% or 50%; token count unchanged, no latency saving', 'Tests whether high-frequency detail is needed only near the objects and the gripper'], 0.5, 3.95, 4.4, 1.3, { size: 11.5, gap: 3 });
  if (WALL) W.pair(s, 'fov'); else { clipPlaceholder(s, 5.2, 1.05, 4.3, 3.05, 'CLIP: original vs foveation', 'same episode, same initial state');
  caption(s, 'Rollout: original (left) and foveated observation (right), success or failure badge at the end.', 5.2, 4.2, 4.3, { size: 10.5, h: 0.5 }); }
  notes(s, N, 11, 'Trick one, visual foveation. The observation keeps a sharp central disc and blurs the periphery. The token count is unchanged, so this changes what the policy sees, not how much it computes.',
    'Left: Fig. 2(a) foveation panel. Right: side-by-side rollout clip.');
}

// ═════════════════════════ 3. ACTION REPEAT (11 s)
{
  const s = S(11);
  title(s, 'Trick 2  Action repeat');
  axisTag(s, 'When to act', 'warn');
  s.addImage({ path: A('m_repeat.png'), x: 0.5, y: 1.05, w: 4.3, h: 2.98 });
  body(s, ['Hold each predicted action for k = 2 or 4 steps, about one call per k steps', 'Cuts policy calls by about k, but lengthens the open-loop interval', 'Fixed reduction, no state check: a speed-oriented baseline'], 0.5, 4.15, 4.4, 1.1, { size: 11.5, gap: 3 });
  if (WALL) W.pair(s, 'repeat'); else { clipPlaceholder(s, 5.2, 1.05, 4.3, 3.05, 'CLIP: original vs action repeat', 'same episode, same initial state');
  caption(s, 'Rollout: watch contact and placement, where a held action can miss a needed correction.', 5.2, 4.2, 4.3, { size: 10.5, h: 0.5 }); }
  notes(s, N, 11, 'Trick two, action repeat. Each predicted action is held for two or four control steps, so the policy is called less often. No state check, a pure speed baseline.',
    'Left: Fig. 2(a) action repeat panel. Right: side-by-side rollout clip.');
}

// ═════════════════════════ 4. DEPTH PRUNING (11 s)
{
  const s = S(12);
  title(s, 'Trick 3  Depth pruning');
  axisTag(s, 'How much to compute', 'teal');
  s.addImage({ path: A('m_prune.png'), x: 0.5, y: 1.05, w: 2.75, h: 2.83 });
  body(s, ['Score each decoder block by Block Influence', 'Remove 1, 2, or 4 low-influence blocks; none adjacent, early blocks and the final block kept', 'Fixed once from calibration data, not adapted at test time'], 3.4, 1.1, 1.5, 3.9, { size: 11, gap: 6 });
  if (WALL) W.pair(s, 'prune'); else { clipPlaceholder(s, 5.2, 1.05, 4.3, 3.05, 'CLIP: original vs depth pruning', 'same episode, same initial state');
  caption(s, 'Rollout: CogACT on WidowX, where pruning removes decoder layers without breaking the closed loop.', 5.2, 4.2, 4.3, { size: 10.5, h: 0.5 }); }
  notes(s, N, 12, 'Trick three, depth pruning. Decoder blocks that barely change their hidden states are removed, one, two, or four of them, keeping the early and final blocks. The selection is fixed before evaluation.',
    'Left: Fig. 2(a) panel. Right: side-by-side rollout clip.');
}

// ═════════════════════════ 5. GUARDED REUSE (11 s)
{
  const s = S(11);
  title(s, 'Trick 4  Guarded action reuse');
  axisTag(s, 'When to act', 'warn');
  s.addImage({ path: A('m_guarded.png'), x: 0.5, y: 1.05, w: 4.3, h: 2.77 });
  body(s, ['Reuse the previous action only when all gates pass: small global and local image change, recent actions agree, translation not near zero, gripper unchanged, reuse count below a cap', 'Presets: strict, moderate, aggressive'], 0.5, 3.95, 4.4, 1.3, { size: 11, gap: 3 });
  if (WALL) W.pair(s, 'reuse'); else { clipPlaceholder(s, 5.2, 1.05, 4.3, 3.05, 'CLIP: original vs guarded reuse', 'same episode, same initial state');
  caption(s, 'Rollout: overlay the gate indicators and a "calls skipped" counter on the right pane.', 5.2, 4.2, 4.3, { size: 10.5, h: 0.5 }); }
  notes(s, N, 11, 'Trick four, guarded reuse. The previous action is reused only while every gate passes: image change, action agreement, translation, and gripper. Any failed gate restores a full call.',
    'Left: Fig. 2(a) guarded reuse panel. Right: side-by-side rollout clip with gate overlay.');
}

// ═════════════════════════ 6. TEMPORAL FUSION (12 s)
{
  const s = S(12);
  title(s, 'Trick 5  Temporal fusion');
  axisTag(s, 'How much to compute', 'teal');
  s.addImage({ path: A('m_fusion.png'), x: 0.5, y: 1.05, w: 2.55, h: 2.08 });
  body(s, ['Reuse visual tokens of stable patches from the previous call', 'Recompute patches with motion, local structure, language attention, or in the interaction region', 'Presets: motion-entropy, task-aware, conservative-adaptive'], 3.2, 1.05, 1.7, 3.6, { size: 11, gap: 6 });
  s.addImage({ path: A('m_cache.png'), x: 0.5, y: 3.22, w: SUB ? 2.35 : 2.55, h: SUB ? 1.44 : 1.56 });
  caption(s, 'Top: fusion mask. Bottom: VLA-Cache criterion.', 0.5, SUB ? 4.7 : 4.9, 2.8, { size: 9.5, h: 0.35 });
  if (WALL) W.pair(s, 'fusion'); else { clipPlaceholder(s, 5.2, 1.05, 4.3, 3.05, 'CLIP: original vs temporal fusion', 'same episode, same initial state');
  caption(s, 'Rollout: optionally tint reused patches on the right pane to show what is carried over between frames.', 5.2, 4.2, 4.3, { size: 10.5, h: 0.5 }); }
  notes(s, N, 12, 'Trick five, temporal fusion. Visual tokens of stable patches are reused from the previous call, while patches with motion, entropy, or language attention are recomputed. The policy still runs every step.',
    'Left: Fig. 2(a) panel. Right: side-by-side rollout clip.');
}

// ═════════════════════════ 7. PROTOCOL (12 s)
{
  const s = S(12);
  title(s, 'Results: matched evaluation protocol');
  const hdr = { bold: true, color: C.white, fill: { color: C.dark }, align: 'center', fontSize: 12 };
  const bb = ['CogACT (7.6B)', 'OpenVLA (7B)', 'SpatialVLA (4B)', 'CronusVLA (0.5B)', 'UniVLA (8.5B)', 'MiniVLA (1B)', 'SmolVLA (0.45B)'];
  const env = [[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1]];
  const rows = [[{ text: 'Backbone', options: hdr }, { text: 'WidowX', options: hdr }, { text: 'Fractal', options: hdr }, { text: 'LIBERO', options: hdr }]];
  bb.forEach((b, i) => rows.push([{ text: b, options: { fontSize: 12, color: C.ink, bold: true } }, ...env[i].map(v => ({ text: v ? '●' : '', options: { align: 'center', color: C.teal, fontSize: 13 } }))]));
  s.addTable(rows, { x: 0.5, y: 1.1, w: 5.0, colW: [2.0, 1.0, 1.0, 1.0], rowH: 0.36, fontFace: FONT, border: { type: 'solid', color: C.line, pt: 0.5 }, fill: { color: C.white }, valign: 'middle' });
  caption(s, 'SimplerEnv WidowX: 4 tasks × 50 episodes.  Google Robot/Fractal: 5 tasks × 50.  LIBERO: Long, Goal, Object, Spatial, each suite a separate pair (3 × 4 = 12), giving 22 pairs in total.', 0.5, 4.15, 5.0, { size: 10.5, h: 0.65 });
  const facts = [['22', 'backbone × environment pairs'], ['13 + 1', 'trick settings + original, per pair'], ['286', 'trick settings (13 × 22), all paired with the original']];
  facts.forEach(([v, l], i) => { stat(s, v, l, 5.7 + i * 1.3, 1.1, 1.25, { size: 30, lsize: 10.5 }); });
  body(s, ['Official checkpoints, no retraining', 'Same task instances and initial states as the original', 'Latency = wall-clock per environment step', 'Exact two-sided McNemar test on paired episodes', '4 × RTX 5090; tables report the best setting per trick'], 5.7, 2.9, 3.8, 2.2, { size: 12, gap: 4 });
  notes(s, N, 12, 'Seven backbones, official checkpoints, on WidowX, Fractal, and LIBERO. Thirteen trick settings per pair plus the original, 286 trick settings, each compared with the original on the same episodes.',
    '286 = 13 trick settings × 22 pairs.');
}

if (WALL) { W.M.walls.forEach(w => W.wall(pres, S, w, N + 1)); }

// ═════════════════════════ 8. Q1 SUCCESS (12 s)
{
  const s = S(12);
  title(s, 'Which tricks preserve success?');
  s.addImage({ path: A('fig1_teaser_nolegend.png'), x: 0.5, y: 1.0, w: 4.0, h: SUB ? 3.6 : 3.97 });
  if (SUB) { legend(s, TRICK_COLORS.slice(0, 3), 0.5, 4.68, 1.4, 9); legend(s, TRICK_COLORS.slice(3), 0.5, 4.9, 1.4, 9); } else { legend(s, TRICK_COLORS.slice(0, 3), 0.5, 5.02, 1.4, 9.5); legend(s, TRICK_COLORS.slice(3), 0.5, 5.27, 1.4, 9.5); }
  qTag(s, 'Q1  Success', 4.9, 1.1);
  s.addText('Guarded reuse and temporal fusion are the safest for preserving success.', { x: 6.35, y: 1.0, w: 3.15, h: 0.65, fontFace: FONT, fontSize: 13.5, bold: true, color: C.ink, valign: 'middle', margin: 0, isTextBox: true });
  caption(s, 'Selected backbones also gain:', 4.9, 1.85, 4.6, { size: 11.5, color: C.ink });
  chip(s, 'Foveation, WidowX: CogACT +2.5 · SpatialVLA +5.0 · CronusVLA +2.5', 4.9, 2.15, 4.6, 'pos', 0.42, 10.5);
  chip(s, 'Depth pruning, WidowX: CogACT +9.5', 4.9, 2.65, 4.6, 'pos', 0.42, 11);
  caption(s, 'Backbone-dependent:', 4.9, 3.2, 4.6, { size: 11.5, color: C.ink });
  chip(s, 'Foveation, LIBERO Goal: UniVLA +7.0  vs  SmolVLA −13.0', 4.9, 3.5, 4.6, 'warn', 0.42, 11);
  chip(s, 'Action repeat: mainly a speed baseline', 4.9, 4.0, 4.6, 'neutral', 0.42, 11);
  caption(s, 'Fig. 1: success per configuration, six WidowX backbones (14 bars each, darker to lighter within a trick). Dotted line = original policy, 0 = no successful episodes. Deltas from Tables I and II, best setting per trick.', 4.9, SUB ? 4.5 : 4.55, 4.6, { size: 9.5, h: SUB ? 0.55 : 0.7 });
  notes(s, N, 12, 'First, success. Guarded reuse and temporal fusion are the safest for preserving success. Foveation and depth pruning help selected backbones. Action repeat is mainly a speed baseline.',
    'Fig. 1 with a native legend. Optionally enlarge one row of panels at a time.');
}

// ═════════════════════════ 9. QUALITATIVE (12 s)
if (!WALL) {
  const s = S(12);
  title(s, 'Depth pruning helps CogACT on WidowX');
  s.addImage({ path: A('fig3a_cogact.png'), x: 0.5, y: 1.0, w: 8.85, h: 2.95 });
  clipPlaceholder(s, 0.5, SUB ? 4.05 : 4.25, 4.4, 0.95, 'CLIP: original vs depth pruning (paired episode)', null);
  chip(s, 'Success 50.0 → 59.5 %', 5.1, SUB ? 4.05 : 4.25, 2.15, 'pos', 0.4, 11.5);
  chip(s, 'Latency 141.4 → 137.5 ms/step', 7.35, SUB ? 4.05 : 4.25, 2.15, 'teal', 0.4, 10.5);
  caption(s, 'Red = failure, green = success. Selected episodes, same task instances and initial states. Aggregate values from Table I.', 5.1, SUB ? 4.52 : 4.75, 4.4, { size: 9.5, h: 0.5 });
  notes(s, N, 12, 'CogACT on WidowX. In these paired episodes the original fails and depth pruning succeeds. Overall success rises from 50.0 to 59.5, and latency drops slightly.',
    'Fig. 3(a) strips. Bottom-left: side-by-side recorded episode.');
}

// ═════════════════════════ 10. SAME TRICK, DIFFERENT BACKBONE (10 s)
if (!WALL) {
  const s = S(10);
  title(s, 'Same trick, same benchmark, different backbone');
  clipPlaceholder(s, 0.5, 1.05, 2.9, 2.6, 'CLIP: MiniVLA original', 'succeeds (green border)');
  clipPlaceholder(s, 3.55, 1.05, 2.9, 2.6, 'CLIP: MiniVLA + pruning', 'fails (red border)');
  caption(s, 'Selected episode, same initial state, WidowX', 0.5, 3.72, 5.95, { size: 10.5 });
  card(s, 6.7, 1.05, 2.8, 2.6);
  s.addText('Depth pruning, WidowX', { x: 6.7, y: 1.15, w: 2.8, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: C.ink, align: 'center', margin: 0, isTextBox: true });
  chip(s, 'CogACT  +9.5', 6.9, 1.6, 2.4, 'pos', 0.5, 14);
  chip(s, 'MiniVLA  −17.5', 6.9, 2.2, 2.4, 'neg', 0.5, 14);
  chip(s, 'SpatialVLA  −6.5', 6.9, 2.8, 2.4, 'neg', 0.5, 14);
  s.addShape('roundRect', { x: 0.5, y: 4.2, w: 9.0, h: 0.85, fill: { color: C.dark }, line: { color: C.dark, width: 0 }, rectRadius: 0.1 });
  s.addText('Structural redundancy is not a property of the trick alone. It depends on how each backbone uses its decoder layers during control.', { x: 0.7, y: 4.2, w: 8.6, h: 0.85, fontFace: FONT, fontSize: 14, bold: true, color: C.darkText, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  notes(s, N, 10, 'Same trick, same benchmark, different backbone. On MiniVLA, depth pruning drops success by 17.5 points, on SpatialVLA by 6.5. Here the pruned policy fails.',
    'Record a MiniVLA WidowX episode from the same task family as the CogACT clip.');
}

// ═════════════════════════ 11. Q2 SPEED (13 s)
{
  const s = S(13);
  title(s, 'Which tricks are actually faster?');
  s.addImage({ path: A('fig4_paper_nolegend.png'), x: 0.4, y: 0.95, w: SUB ? 4.35 : 4.7, h: SUB ? 3.56 : 3.85 });
  if (SUB) { legend(s, TRICK_COLORS.slice(0, 3), 0.5, 4.62, 1.5, 9); legend(s, TRICK_COLORS.slice(3), 0.5, 4.84, 1.5, 9); } else { legend(s, TRICK_COLORS.slice(0, 3), 0.5, 4.85, 1.55, 9.5); legend(s, TRICK_COLORS.slice(3), 0.5, 5.1, 1.55, 9.5); }
  caption(s, 'Fig. 4, SimplerEnv WidowX. Star = original, shaded = faster and no worse. Colour = trick; circle, square, triangle = first, second, third setting of that trick.', 5.4, 0.95, 4.1, { size: 10, h: 0.7 });
  qTag(s, 'Q2  Speed', 5.4, 1.75);
  s.addText('Action repeat gives the largest speedup, depth pruning gives decoder-level savings, guarded reuse gives the safest trade-off.', { x: 5.4, y: 2.25, w: 4.1, h: 0.9, fontFace: FONT, fontSize: 12, bold: true, color: C.ink, valign: 'top', margin: 0, isTextBox: true });
  const rows = [
    [{ text: 'Latency, ms per step', options: { bold: true, color: C.white, fill: { color: C.dark } } }, { text: 'Original → trick', options: { bold: true, color: C.white, fill: { color: C.dark }, align: 'center' } }],
    ['Repeat · SpatialVLA WidowX', { text: '423.5 → 235.1', options: { align: 'center', bold: true, color: C.orange } }],
    ['Repeat · OpenVLA LIBERO Long', { text: '171.6 → 93.2', options: { align: 'center', bold: true, color: C.orange } }],
    ['Pruning · OpenVLA WidowX', { text: '210.2 → 199.4', options: { align: 'center', bold: true, color: C.teal } }],
    ['Reuse · SmolVLA LIBERO Long', { text: '289.5 → 268.9', options: { align: 'center', bold: true, color: C.teal } }],
  ];
  s.addTable(rows, { x: 5.4, y: 3.35, w: 4.1, colW: [2.55, 1.55], rowH: 0.3, fontFace: FONT, fontSize: 10, color: C.ink, border: { type: 'solid', color: C.line, pt: 0.5 }, valign: 'middle', margin: 0.03 });
  notes(s, N, 13, 'Second, speed. Action repeat gives the largest speedup. SpatialVLA on WidowX drops from about 424 to 235 milliseconds per step. Depth pruning gives smaller decoder-level savings, and guarded reuse the safest trade-off.',
    'Paper Fig. 4 with a native colour legend. If the Speechma clip runs longer than 13 s, extend the slide rather than speeding the voice.');
}

// ═════════════════════════ 12. OPEN LOOP VS GATING (13 s)
if (!WALL) {
  const s = S(13);
  title(s, 'Open loop has a cost, gating keeps it safer');
  const labels = ['Original', 'Action repeat', 'Guarded reuse'];
  const subs = ['policy call every step', 'held actions, episode fails', 'calls skipped only when the gates pass'];
  labels.forEach((l, i) => { clipPlaceholder(s, 0.5 + i * 3.05, 1.05, 2.9, 2.15, 'CLIP: ' + l, subs[i]); caption(s, 'policy calls used: ___', 0.5 + i * 3.05, 3.25, 2.9, { size: 10, align: 'center' }); });
  chip(s, 'Action repeat (k = 2, best setting), WidowX: CogACT −38.0 · UniVLA −75.0', 0.5, 3.7, 4.4, 'neg', 0.5, 10.5);
  chip(s, 'Fractal (milder): CogACT −1.2 · SpatialVLA 0.0', 0.5, 4.3, 4.4, 'neutral', 0.5, 11);
  chip(s, 'Guarded reuse, SmolVLA LIBERO Long: 289.5 → 268.9 ms', 5.1, 3.7, 4.4, 'teal', 0.5, 11);
  chip(s, 'success unchanged at 42.0 %', 5.1, 4.3, 4.4, 'pos', 0.5, 11);
  caption(s, 'Same CogACT WidowX episode and initial state in all three panels. Deltas from Tables I and II, best setting per trick.', 0.5, SUB ? 4.82 : 4.85, 9.0, { size: 10, h: 0.26 });
  notes(s, N, 13, 'Action repeat runs open loop: on WidowX, CogACT loses 38 points and UniVLA 75. Guarded reuse skips calls only when the gates agree. On SmolVLA in LIBERO Long, success stays at 42.0 with lower latency.',
    'Three-panel recording of one CogACT WidowX episode; use k = 2 for the action-repeat panel so the clip matches the quoted deltas (k = 4 is worse, about -46 and -85 points, Fig. 5). Before rendering, fill the three "policy calls used" counters from that episode: original = number of steps, repeat = about steps / k, guarded = steps minus skipped calls, or delete the counter lines.');
}

// ═════════════════════════ 13. Q3 TRANSFER (12 s)
{
  const s = S(12);
  title(s, 'Does the effect transfer across models?');
  s.addImage({ path: A('consistency.png'), x: 0.5, y: 1.0, w: 5.6, h: 2.76 });
  caption(s, 'For each setting: how many of 13 backbone and environment pairs got lower, unchanged, or higher success. LIBERO episodes are pooled over the four suites, so 13 pairs stand for the 22.', 0.5, 3.8, 5.6, { size: 10, h: 0.5 });
  qTag(s, 'Q3  Transfer', 6.4, 1.1);
  s.addText('Effects are a joint property of the trick, the backbone, and the task distribution.', { x: 6.4, y: 1.55, w: 3.1, h: 0.55, fontFace: FONT, fontSize: 12, bold: true, color: C.ink, valign: 'middle', margin: 0, isTextBox: true });
  chip(s, 'Depth pruning, WidowX: CogACT +9.5 · MiniVLA −17.5', 6.4, 2.2, 3.1, 'warn', 0.55, 10.5);
  chip(s, 'Foveation, LIBERO Goal: UniVLA +7.0 · SmolVLA −13.0', 6.4, 2.85, 3.1, 'warn', 0.55, 10.5);
  chip(s, 'Action repeat: often harmful on WidowX, milder on Fractal for some backbones', 6.4, 3.5, 3.1, 'warn', 0.62, 10.5);
  s.addShape('roundRect', { x: 0.5, y: 4.45, w: 9.0, h: 0.6, fill: { color: C.dark }, line: { color: C.dark, width: 0 }, rectRadius: 0.1 });
  s.addText('No trick is plug-and-play: each one needs matched evaluation under the target backbone and environment.', { x: 0.6, y: 4.45, w: 8.8, h: 0.6, fontFace: FONT, fontSize: 14, bold: true, color: C.darkText, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  notes(s, N, 12, 'Third, consistency. Across backbone and environment pairs the same trick improves one model, leaves another unchanged, and hurts a third. No trick is plug-and-play.',
    'Consistency chart (supplementary figure, counts from the per-setting records with LIBERO episodes pooled). Do not read the counts aloud.');
}

// ═════════════════════════ 14. TAKEAWAYS (12 s)
{
  const s = S(12);
  title(s, 'Takeaways');
  const v = [
    ['Guarded reuse', 'Most reliable. Preserves success, modest speedup when the gates open.', C.greenSoft, C.green],
    ['Action repeat', 'Largest speedup, but the risk of long open-loop execution.', C.orangeSoft, C.orange],
    ['Depth pruning', 'Decoder-level savings, benefit depends on the backbone.', C.tealSoft, C.teal],
    ['Foveation · Temporal fusion', 'Test how much the visual representation can change while the policy stays stable.', C.blueSoft, C.blue],
  ];
  v.forEach(([h, t, soft, col], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.1 + Math.floor(i / 2) * 1.5;
    card(s, x, y, 4.4, 1.3, { fill: soft, line: soft });
    s.addText(h, { x: x + 0.2, y: y + 0.12, w: 4.0, h: 0.4, fontFace: HFONT, fontSize: 16, bold: true, color: col, margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.2, y: y + 0.52, w: 4.0, h: 0.7, fontFace: FONT, fontSize: 12.5, color: C.ink, valign: 'top', margin: 0, isTextBox: true });
  });
  s.addShape('roundRect', { x: 0.5, y: 4.25, w: 9.0, h: 0.8, fill: { color: C.dark }, line: { color: C.dark, width: 0 }, rectRadius: 0.1 });
  s.addText('VLA efficiency is policy- and environment-dependent. Evaluate every trick under the target backbone and environment before applying it.', { x: 0.7, y: 4.25, w: 8.6, h: 0.8, fontFace: FONT, fontSize: 14, bold: true, color: C.darkText, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  notes(s, N, 12, 'Guarded reuse is most reliable. Action repeat is fastest but risky. Depth pruning is backbone-dependent. Evaluate every trick under the target backbone and environment before applying it. Thank you.',
    'Final slide. No names, logos, or URLs. Hold 2 s after the narration ends, then fade to black.');
}

const out = require('path').join(__dirname, WALL ? (SUB ? 'ICRA27_video_v3_wall_subs.pptx' : 'ICRA27_video_v3_wall.pptx') : (SUB ? 'ICRA27_video_v2_subs.pptx' : 'ICRA27_video_v2.pptx'));
require('fs').writeFileSync(out.replace('.pptx', '.secs.json'), JSON.stringify(SECS));
pres.writeFile({ fileName: out }).then(() => console.log('wrote', out, 'slides', N));
