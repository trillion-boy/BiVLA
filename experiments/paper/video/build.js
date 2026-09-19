const pptxgen = require('pptxgenjs');
const L = require('./lib');
const { A, C, FONT, HFONT, base, title, timeTag, caption, body, card, badge, clipPlaceholder, notes, stat } = L;

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9'; // 10 x 5.625 in
pres.title = 'Bag of Tricks for Training-Free VLA Models, accompanying video storyboard';

let N = 0;
const S = (secs, dark = false) => { const s = pres.addSlide(); N += 1; base(s, dark); s.__n = N; s.__secs = secs; timeTag(s, N, secs, dark); return s; };
const chip = (s, text, x, y, w, kind = 'neutral', h = 0.42, size = 13) => {
  const map = { pos: [C.greenSoft, C.green], neg: [C.redSoft, C.red], neutral: [C.card, C.ink], teal: [C.tealSoft, C.teal], warn: [C.orangeSoft, C.orange] };
  const [fill, col] = map[kind];
  s.addShape('roundRect', { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.1 });
  s.addText(text, { x, y, w, h, fontFace: FONT, fontSize: size, bold: true, color: col, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
};

// ───────────────────────── 1. Title (7 s)
{
  const s = S(7, true);
  s.addImage({ path: A('fig2_overview.png'), x: 0.6, y: 1.0, w: 8.8, h: 3.78, transparency: 88 });
  s.addText('Bag of Tricks for Training-Free\nVision-Language-Action Models', { x: 0.7, y: 1.35, w: 8.6, h: 1.5, fontFace: HFONT, fontSize: 34, bold: true, color: C.darkText, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  s.addText('What to See, When to Act, and How Much to Compute?', { x: 0.7, y: 2.95, w: 8.6, h: 0.55, fontFace: FONT, fontSize: 20, italic: true, color: 'C9D3E0', align: 'center', margin: 0, isTextBox: true });
  s.addText('Anonymous submission  ·  ICRA 2027 accompanying video', { x: 0.7, y: 4.25, w: 8.6, h: 0.4, fontFace: FONT, fontSize: 14, color: '9AA7B8', align: 'center', margin: 0, isTextBox: true });
  notes(s, N, 7, 'This video accompanies an anonymous submission on training-free efficiency tricks for vision-language-action models.',
    'Static title card. No names, affiliations, logos, or URLs anywhere in the video.');
}

// ───────────────────────── 2. Headline (10 s)
{
  const s = S(10);
  title(s, 'No trick is plug-and-play');
  caption(s, 'Depth pruning on SimplerEnv WidowX, success change in points, three backbones.', 0.5, 1.05, 9, { size: 14, color: C.ink });
  const xs = [0.5, 3.75, 7.0];
  const items = [['CogACT', '+9.5', 'pos', '50.0 → 59.5'], ['MiniVLA', '−17.5', 'neg', '36.0 → 18.5'], ['SpatialVLA', '−6.5', 'neg', '45.0 → 38.5']];
  items.forEach(([name, d, k, rng], i) => {
    card(s, xs[i], 1.65, 2.5, 2.55);
    s.addText(name, { x: xs[i], y: 1.8, w: 2.5, h: 0.4, fontFace: HFONT, fontSize: 18, bold: true, color: C.ink, align: 'center', margin: 0, isTextBox: true });
    s.addText(d, { x: xs[i], y: 2.25, w: 2.5, h: 1.0, fontFace: HFONT, fontSize: 54, bold: true, color: k === 'pos' ? C.green : C.red, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    s.addText(rng + ' %', { x: xs[i], y: 3.3, w: 2.5, h: 0.35, fontFace: FONT, fontSize: 13, color: C.muted, align: 'center', margin: 0, isTextBox: true });
    s.addText('success', { x: xs[i], y: 3.62, w: 2.5, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, align: 'center', margin: 0, isTextBox: true });
  });
  s.addText('The same trick helps one model, leaves another unchanged, and hurts a third.', { x: 0.5, y: 4.45, w: 9, h: 0.45, fontFace: FONT, fontSize: 16, bold: true, color: C.teal, align: 'center', margin: 0, isTextBox: true });
  caption(s, 'Best setting per trick, Table I of the paper.', 0.5, 5.0, 6, { size: 10 });
  notes(s, N, 10, 'Our headline finding. The same training-free trick, on the same benchmark, helps one model and hurts two others. No trick is plug-and-play.',
    'Numbers animate in one by one (CogACT, then MiniVLA, then SpatialVLA). All three from Table I, WidowX, depth pruning row.');
}

// ───────────────────────── 3. Problem (10 s)
{
  const s = S(10);
  title(s, 'VLA models are strong but slow');
  clipPlaceholder(s, 0.5, 1.1, 5.2, 3.6, 'CLIP: one original WidowX rollout', 'Overlay a live counter "ms per step" (e.g. CogACT 141 ms, SpatialVLA 424 ms).\nRecord from the evaluation run, 720p, 8-10 s.');
  const stack = [['Image + instruction', C.card], ['Vision encoder', C.blueSoft], ['Language backbone', C.blueSoft], ['Action', C.card]];
  stack.forEach(([t, f], i) => {
    const y = 1.1 + i * 0.62;
    s.addShape('roundRect', { x: 6.1, y, w: 3.4, h: 0.48, fill: { color: f }, line: { color: C.line, width: 0.75 }, rectRadius: 0.08 });
    s.addText(t, { x: 6.1, y, w: 3.4, h: 0.48, fontFace: FONT, fontSize: 14, bold: i === 1 || i === 2, color: C.ink, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    if (i < 3) s.addText('▼', { x: 7.65, y: y + 0.46, w: 0.3, h: 0.18, fontFace: FONT, fontSize: 8, color: C.muted, align: 'center', margin: 0, isTextBox: true });
  });
  body(s, ['Every control step runs a large vision encoder and language backbone', 'Existing tricks are each reported for one trick, one backbone, one benchmark', 'Do the gains transfer?'], 6.1, 3.7, 3.4, 1.4, { size: 12.5, gap: 4 });
  notes(s, N, 10, 'VLA models are capable but slow. Each control step runs a vision encoder and language backbone. Existing tricks are each reported in one setting.',
    'Left: recorded original-policy episode with a latency counter burned in. Right: static pipeline stack.');
}

// ───────────────────────── 4. Three axes (12 s)
{
  const s = S(12);
  title(s, 'Five tricks, three axes of the control loop', { size: 24 });
  s.addImage({ path: A('fig2_overview.png'), x: 0.5, y: 1.0, w: 9.0, h: 3.87 });
  const ax = [['What to see', 'Foveation', C.blue, C.blueSoft], ['When to act', 'Action repeat · Guarded reuse', C.orange, C.orangeSoft], ['How much to compute', 'Depth pruning · Temporal fusion', C.teal, C.tealSoft]];
  ax.forEach(([h, t, col, soft], i) => {
    const x = 0.5 + i * 3.05;
    s.addShape('roundRect', { x, y: 4.95, w: 2.9, h: 0.5, fill: { color: soft }, line: { color: soft, width: 0 }, rectRadius: 0.08 });
    s.addText([{ text: h + '  ', options: { bold: true, color: col } }, { text: t, options: { color: C.ink } }], { x, y: 4.95, w: 2.9, h: 0.5, fontFace: FONT, fontSize: 11.5, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  });
  notes(s, N, 12, 'We study five training-free tricks along three axes of the control loop. What to see, when to act, and how much to compute. Each trick changes both cost and behavior.',
    'Fig. 2(a) of the paper. Highlight each trick box in sequence as it is named. The original policy is the unmodified loop and counts as one of six variants.');
}

// ───────────────────────── 5. What to see (8 s)
{
  const s = S(8);
  title(s, 'What to see: visual foveation');
  s.addImage({ path: A('ov_fov.png'), x: 0.5, y: 1.1, w: 4.3, h: 1.45 });
  body(s, ['Sharp central disc, progressively blurred periphery', 'Keep ratio 20% or 50% of the image', 'Token count unchanged, so this changes information, not FLOPs'], 0.5, 2.85, 4.3, 1.9, { size: 14, gap: 6 });
  const fx = [5.3, 6.75, 8.2]; const fl = ['Original frame', 'Keep 20%', 'Keep 50%']; const fi = ['frame_raw.png', 'frame_fov20.png', 'frame_fov50.png'];
  fi.forEach((f, i) => {
    s.addImage({ path: A(f), x: fx[i], y: 1.5, w: 1.3, h: 0.92 });
    caption(s, fl[i], fx[i], 2.47, 1.3, { align: 'center', size: 11 });
  });
  card(s, 5.3, 3.1, 4.2, 1.6, { fill: C.blueSoft, line: C.blueSoft });
  s.addText('Tests whether a policy needs high-frequency detail only near the objects and the gripper.', { x: 5.45, y: 3.2, w: 3.9, h: 1.4, fontFace: FONT, fontSize: 13, color: C.ink, valign: 'middle', margin: 0, isTextBox: true });
  notes(s, N, 8, 'What to see. Visual foveation keeps a sharp central disc and blurs the periphery. The token count is unchanged.',
    'The three frames are a WidowX observation rendered with the foveation mixture at keep ratio 20% and 50% (replace with frames from the actual foveation code if preferred).');
}

// ───────────────────────── 6. When to act (10 s)
{
  const s = S(10);
  title(s, 'When to act: action repeat vs guarded reuse', { size: 24 });
  const rows = [
    ['Original', 'policy call every step', Array(12).fill(1)],
    ['Action repeat k = 4', 'one call, then three held actions', [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]],
    ['Guarded reuse', 'skip only while every gate passes', [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1]],
  ];
  rows.forEach(([name, sub, dots], r) => {
    const y = 1.2 + r * 0.95;
    s.addText(name, { x: 0.5, y, w: 2.6, h: 0.3, fontFace: FONT, fontSize: 14, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(sub, { x: 0.5, y: y + 0.3, w: 2.6, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, margin: 0, isTextBox: true });
    dots.forEach((d, i) => {
      const x = 3.3 + i * 0.5;
      s.addShape('ellipse', { x, y: y + 0.12, w: 0.3, h: 0.3, fill: { color: d ? (r === 0 ? C.ink : C.orange) : C.white }, line: { color: d ? (r === 0 ? C.ink : C.orange) : '9AA3AE', width: 1.25 } });
    });
    if (r === 2) dots.forEach((d, i) => { if (!d) s.addShape('ellipse', { x: 3.3 + i * 0.5 + 0.09, y: y + 0.55, w: 0.12, h: 0.12, fill: { color: C.green }, line: { color: C.green, width: 0 } }); });
  });
  caption(s, 'filled = policy call    hollow = previous action held    green dot = all gates passed', 3.3, 3.95, 6.2, { size: 10.5 });
  card(s, 0.5, 4.3, 9.0, 0.85, { fill: C.orangeSoft, line: C.orangeSoft });
  s.addText([{ text: 'Gates: ', options: { bold: true, color: C.orange } }, { text: 'global and local image change small · recent actions agree · translation not near zero · gripper unchanged · reuse count below cap.   Presets: strict, moderate, aggressive.', options: { color: C.ink } }],
    { x: 0.65, y: 4.35, w: 8.7, h: 0.75, fontFace: FONT, fontSize: 12, valign: 'middle', margin: 0, isTextBox: true });
  notes(s, N, 10, 'When to act. Action repeat holds each action for two or four steps. Guarded reuse skips the policy call only when every gate passes.',
    'Timeline of policy calls. Optionally animate the dots left to right in step with the narration.');
}

// ───────────────────────── 7. How much to compute (10 s)
{
  const s = S(10);
  title(s, 'How much to compute: pruning and fusion', { size: 24 });
  s.addImage({ path: A('ov_prune.png'), x: 0.5, y: 1.1, w: 4.3, h: 1.8 });
  body(s, ['Remove 1, 2, or 4 decoder layers with the lowest Block Influence', 'None adjacent, first and last layers kept', 'Selected once from calibration data, fixed before evaluation'], 0.5, 3.1, 4.3, 1.9, { size: 13, gap: 5 });
  s.addImage({ path: A('ov_fusion.png'), x: 5.2, y: 1.1, w: 4.3, h: 1.8 });
  body(s, ['Reuse projected visual tokens of stable patches from the previous call', 'Protect patches with motion, entropy, language attention, or interaction regions', 'Presets: motion-entropy, task-aware, conservative-adaptive'], 5.2, 3.1, 4.3, 1.9, { size: 13, gap: 5 });
  notes(s, N, 10, 'How much to compute. Depth pruning removes one, two, or four decoder layers with lowest Block Influence. Temporal fusion reuses visual tokens of stable patches.',
    'Two crops from Fig. 2(a). Optional: fade out the pruned layers in the left diagram.');
}

// ───────────────────────── 8. Protocol (12 s)
{
  const s = S(12);
  title(s, 'Matched evaluation protocol');
  const hdr = { bold: true, color: C.white, fill: { color: C.dark }, align: 'center', fontSize: 12 };
  const bb = ['CogACT (7.6B)', 'OpenVLA (7B)', 'SpatialVLA (4B)', 'CronusVLA (0.5B)', 'UniVLA (8.5B)', 'MiniVLA (1B)', 'SmolVLA (0.45B)'];
  const env = [[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1]];
  const rows = [[{ text: 'Backbone', options: hdr }, { text: 'WidowX', options: hdr }, { text: 'Fractal', options: hdr }, { text: 'LIBERO', options: hdr }]];
  bb.forEach((b, i) => rows.push([{ text: b, options: { fontSize: 12, color: C.ink, bold: true } }, ...env[i].map(v => ({ text: v ? '●' : '', options: { align: 'center', color: C.teal, fontSize: 13 } }))]));
  s.addTable(rows, { x: 0.5, y: 1.1, w: 5.0, colW: [2.0, 1.0, 1.0, 1.0], rowH: 0.36, fontFace: FONT, border: { type: 'solid', color: C.line, pt: 0.5 }, fill: { color: C.white }, valign: 'middle' });
  caption(s, 'SimplerEnv WidowX: 4 tasks × 50 episodes.  Google Robot/Fractal: 5 tasks × 50.  LIBERO: Long, Goal, Object, Spatial.', 0.5, 4.15, 5.0, { size: 10.5, h: 0.5 });
  const facts = [['22', 'backbone × environment pairs'], ['13 + 1', 'trick settings + original, per pair'], ['286', 'trick settings, all paired with the original']];
  facts.forEach(([v, l], i) => { stat(s, v, l, 5.7 + i * 1.3, 1.1, 1.25, { size: 30, lsize: 10.5 }); });
  body(s, ['Official checkpoints, no retraining', 'Same task instances and initial states as the original', 'Latency = wall-clock per environment step', 'Exact two-sided McNemar test on paired episodes', '4 × RTX 5090; tables report the best setting per trick'], 5.7, 2.9, 3.8, 2.2, { size: 12, gap: 4 });
  notes(s, N, 12, 'Seven open-source backbones, official checkpoints, no retraining, on SimplerEnv WidowX, Fractal, and LIBERO. Thirteen trick settings per pair, 286 settings in total, same initial states, paired McNemar tests.',
    '286 = 13 trick settings × 22 pairs (the original policy is the 14th configuration of each pair).');
}

// ───────────────────────── 9. RQ-1 success (12 s)
{
  const s = S(12);
  title(s, 'RQ-1  Which tricks preserve success?', { size: 24 });
  s.addImage({ path: A('fig1_teaser.png'), x: 0.5, y: 1.05, w: 4.6, h: 4.14 });
  badge(s, 'A-1', 5.5, 1.15, { d: 0.5, size: 12 });
  s.addText('Guarded reuse and temporal fusion match or improve success for most backbones.', { x: 6.1, y: 1.1, w: 3.4, h: 0.65, fontFace: FONT, fontSize: 14, bold: true, color: C.ink, valign: 'middle', margin: 0, isTextBox: true });
  caption(s, 'Selected backbones also gain:', 5.5, 1.95, 4.0, { size: 12, color: C.ink });
  chip(s, 'Foveation, WidowX: CogACT +2.5 · SpatialVLA +5.0 · CronusVLA +2.5', 5.5, 2.3, 4.0, 'pos', 0.5, 10.5);
  chip(s, 'Depth pruning, WidowX: CogACT +9.5', 5.5, 2.9, 4.0, 'pos', 0.42, 11);
  chip(s, 'Foveation, LIBERO Goal: UniVLA +7.0  vs  SmolVLA −13.0', 5.5, 3.42, 4.0, 'warn', 0.42, 11);
  chip(s, 'Action repeat: mainly a speed baseline', 5.5, 3.94, 4.0, 'neutral', 0.42, 11);
  caption(s, 'Fig. 1: success per configuration, six WidowX backbones. Dotted line = original policy. Deltas from Tables I and II, best setting per trick.', 5.5, 4.5, 4.0, { size: 10, h: 0.6 });
  notes(s, N, 12, 'First, success. Guarded reuse and temporal fusion match or improve success for most backbones. Foveation and depth pruning help selected backbones. Action repeat is mainly a speed baseline.',
    'Optionally enlarge one row of the 2×3 panel at a time. Outline guarded-reuse and temporal-fusion bars in green.');
}

// ───────────────────────── 10. Qualitative CogACT (12 s)
{
  const s = S(12);
  title(s, 'Depth pruning helps CogACT on WidowX', { size: 24 });
  s.addImage({ path: A('fig3a_cogact.png'), x: 0.5, y: 1.0, w: 9.0, h: 3.0 });
  clipPlaceholder(s, 0.5, 4.15, 4.4, 1.0, 'CLIP: original vs depth pruning (paired episode)', null);
  chip(s, 'Success 50.0 → 59.5 %', 5.1, 4.15, 2.15, 'pos', 0.4, 11.5);
  chip(s, 'Latency 141.4 → 137.5 ms/step', 7.35, 4.15, 2.15, 'teal', 0.4, 10.5);
  caption(s, 'Red = failure, green = success. Selected episodes, same task instances and initial states. Aggregate values from Table I.', 5.1, 4.65, 4.4, { size: 9.5, h: 0.5 });
  notes(s, N, 12, 'CogACT on WidowX. In these paired episodes the original fails and depth pruning succeeds. Overall success rises from 50.0 to 59.5, and latency drops slightly.',
    'Fig. 3(a) strips (Project Page mention cropped off). Bottom-left: side-by-side recorded episode, original left, depth pruning right, success/fail badge at the end.');
}

// ───────────────────────── 11. Same trick, different backbone (10 s)
{
  const s = S(10);
  title(s, 'Same trick, same benchmark, different backbone', { size: 24 });
  clipPlaceholder(s, 0.5, 1.05, 2.9, 2.2, 'CLIP: MiniVLA original', 'succeeds (green border)');
  clipPlaceholder(s, 3.55, 1.05, 2.9, 2.2, 'CLIP: MiniVLA + pruning', 'fails (red border)');
  caption(s, 'Selected episode, same initial state, WidowX', 0.5, 3.3, 5.95, { size: 10.5 });
  card(s, 6.7, 1.05, 2.8, 2.55);
  s.addText('Depth pruning, WidowX', { x: 6.7, y: 1.15, w: 2.8, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: C.ink, align: 'center', margin: 0, isTextBox: true });
  chip(s, 'CogACT  +9.5', 6.9, 1.6, 2.4, 'pos', 0.5, 14);
  chip(s, 'MiniVLA  −17.5', 6.9, 2.2, 2.4, 'neg', 0.5, 14);
  chip(s, 'SpatialVLA  −6.5', 6.9, 2.8, 2.4, 'neg', 0.5, 14);
  s.addText('Structural redundancy is not a property of the trick alone. It depends on how each backbone uses its decoder layers during control.', { x: 0.5, y: 3.85, w: 9.0, h: 0.9, fontFace: FONT, fontSize: 15, color: C.ink, valign: 'middle', margin: 0, isTextBox: true });
  notes(s, N, 10, 'Same trick, same benchmark, different backbone. On MiniVLA, depth pruning drops success by 17.5 points, on SpatialVLA by 6.5. Here the pruned policy fails.',
    'Record a MiniVLA WidowX episode from the same task family as slide 10 (e.g. carrot on plate) for original and depth pruning.');
}

// ───────────────────────── 12. RQ-2 speed (12 s)
{
  const s = S(12);
  title(s, 'RQ-2  Which tricks are actually faster?', { size: 24 });
  s.addImage({ path: A('fig4_tradeoff_row.png'), x: 0.65, y: 0.95, w: 8.7, h: 2.96 });
  badge(s, 'A-2', 0.5, 4.05, { d: 0.45, size: 11 });
  s.addText('Action repeat gives the largest speedup, depth pruning gives practical decoder-level savings, guarded reuse gives the safest trade-off.', { x: 1.05, y: 3.95, w: 4.2, h: 1.2, fontFace: FONT, fontSize: 12, bold: true, color: C.ink, valign: 'middle', margin: 0, isTextBox: true });
  const rows = [
    [{ text: 'Latency, ms per step', options: { bold: true, color: C.white, fill: { color: C.dark } } }, { text: 'Original → trick', options: { bold: true, color: C.white, fill: { color: C.dark }, align: 'center' } }],
    ['Repeat · SpatialVLA WidowX', { text: '423.5 → 235.1', options: { align: 'center', bold: true, color: C.orange } }],
    ['Repeat · OpenVLA LIBERO Long', { text: '171.6 → 93.2', options: { align: 'center', bold: true, color: C.orange } }],
    ['Pruning · OpenVLA WidowX', { text: '210.2 → 199.4', options: { align: 'center', bold: true, color: C.teal } }],
    ['Reuse · SmolVLA LIBERO Long', { text: '289.5 → 268.9', options: { align: 'center', bold: true, color: C.teal } }],
  ];
  s.addTable(rows, { x: 5.5, y: 3.95, w: 4.0, colW: [2.5, 1.5], rowH: 0.24, fontFace: FONT, fontSize: 9.5, color: C.ink, border: { type: 'solid', color: C.line, pt: 0.5 }, valign: 'middle', margin: 0.03 });
  notes(s, N, 12, 'Second, speed. Action repeat gives the largest speedup. SpatialVLA on WidowX drops from 423.5 to 235.1 milliseconds per step. Depth pruning gives smaller decoder-level savings.',
    'Fig. 4: speedup vs success change, star = original, shaded = faster and no worse. Highlight the action-repeat points at the far right and the depth-pruning points near the star.');
}

// ───────────────────────── 13. Repeat vs reuse in one episode (14 s)
{
  const s = S(14);
  title(s, 'Open loop has a cost, gating keeps it safe', { size: 24 });
  const labels = ['Original', 'Action repeat k = 4', 'Guarded reuse'];
  const subs = ['policy call every step', 'held actions, episode fails', 'gates shown, calls skipped only when they pass'];
  labels.forEach((l, i) => { clipPlaceholder(s, 0.5 + i * 3.05, 1.05, 2.9, 2.15, 'CLIP: ' + l, subs[i]); caption(s, 'policy calls used: ___', 0.5 + i * 3.05, 3.25, 2.9, { size: 10, align: 'center' }); });
  chip(s, 'Action repeat, WidowX: CogACT −38.0 · UniVLA −75.0', 0.5, 3.7, 4.4, 'neg', 0.5, 11);
  chip(s, 'Fractal, some backbones: CogACT −1.2 · SpatialVLA 0.0', 0.5, 4.3, 4.4, 'neutral', 0.5, 11);
  chip(s, 'Guarded reuse, SmolVLA LIBERO Long: 289.5 → 268.9 ms', 5.1, 3.7, 4.4, 'teal', 0.5, 11);
  chip(s, 'success unchanged at 42.0 %', 5.1, 4.3, 4.4, 'pos', 0.5, 11);
  caption(s, 'Same CogACT WidowX episode and initial state in all three panels. Deltas from Tables I and II, best setting per trick.', 0.5, 4.85, 9.0, { size: 10 });
  notes(s, N, 14, 'Action repeat runs open loop. On WidowX, CogACT loses 38 points and UniVLA 75. Guarded reuse skips calls only when the gates agree. SmolVLA on LIBERO Long saves 20 milliseconds per step with success unchanged.',
    'Three-panel recording of one CogACT WidowX episode: original, action repeat k=4, guarded reuse. Under each panel a "policy calls used" counter; in the third panel a five-gate strip that turns green on skipped steps.');
}

// ───────────────────────── 14. RQ-3 consistency (12 s)
{
  const s = S(12);
  title(s, 'RQ-3  Does the effect transfer across models?', { size: 24 });
  s.addImage({ path: A('consistency.png'), x: 0.5, y: 1.0, w: 5.6, h: 2.76 });
  caption(s, 'For each setting: how many of 13 backbone and environment pairs got lower, unchanged, or higher success (LIBERO suites pooled, so 13 stand for the 22 pairs).', 0.5, 3.8, 5.6, { size: 10, h: 0.5 });
  badge(s, 'A-3', 6.4, 1.1, { d: 0.5, size: 12 });
  s.addText('Effects are a joint property of the trick, the backbone, and the task distribution.', { x: 7.0, y: 1.05, w: 2.5, h: 0.9, fontFace: FONT, fontSize: 13, bold: true, color: C.ink, valign: 'middle', margin: 0, isTextBox: true });
  chip(s, 'Depth pruning, WidowX: CogACT +9.5 · MiniVLA −17.5', 6.4, 2.15, 3.1, 'warn', 0.55, 10.5);
  chip(s, 'Foveation, LIBERO Goal: UniVLA +7.0 · SmolVLA −13.0', 6.4, 2.8, 3.1, 'warn', 0.55, 10.5);
  chip(s, 'Action repeat: harmful on WidowX, milder on Fractal for some backbones', 6.4, 3.45, 3.1, 'warn', 0.62, 10.5);
  s.addShape('roundRect', { x: 0.5, y: 4.45, w: 9.0, h: 0.6, fill: { color: C.dark }, line: { color: C.dark, width: 0 }, rectRadius: 0.1 });
  s.addText('Not plug-and-play: each trick needs matched evaluation under the target backbone and environment.', { x: 0.6, y: 4.45, w: 8.8, h: 0.6, fontFace: FONT, fontSize: 14, bold: true, color: C.darkText, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  notes(s, N, 12, 'Third, consistency. Across backbone and environment pairs the same trick improves one model, leaves another unchanged, and hurts a third. Gains depend on the backbone and the environment.',
    'Consistency chart (Project Page figure). Do not read the counts aloud. Optionally highlight the depth-pruning and foveation rows.');
}

// ───────────────────────── 15. Takeaway (10 s)
{
  const s = S(10);
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
  notes(s, N, 10, 'Guarded reuse is most reliable. Action repeat is fastest but risky. Depth pruning is backbone-dependent. Evaluate every trick under the target backbone and environment before applying it.',
    'Four verdict cards in the paper\'s conclusion wording.');
}

// ───────────────────────── 16. End card (4 s)
{
  const s = S(4, true);
  s.addImage({ path: A('fig1_teaser.png'), x: 2.9, y: 0.6, w: 4.2, h: 3.78, transparency: 88 });
  s.addText('Bag of Tricks for Training-Free Vision-Language-Action Models', { x: 0.7, y: 1.9, w: 8.6, h: 0.8, fontFace: HFONT, fontSize: 24, bold: true, color: C.darkText, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  s.addText('Details, per-cell p-values, and all 286 settings are in the paper and its supplementary material.', { x: 0.7, y: 2.8, w: 8.6, h: 0.5, fontFace: FONT, fontSize: 14, color: 'C9D3E0', align: 'center', margin: 0, isTextBox: true });
  s.addText('Anonymous submission  ·  ICRA 2027', { x: 0.7, y: 3.5, w: 8.6, h: 0.4, fontFace: FONT, fontSize: 13, color: '9AA7B8', align: 'center', margin: 0, isTextBox: true });
  notes(s, N, 4, 'Details are in the paper. Thank you.', 'No URLs, names, or logos. Hold 4 s and fade to black.');
}

const out = require('path').join(__dirname, 'ICRA27_video_storyboard.pptx');
pres.writeFile({ fileName: out }).then(() => console.log('wrote', out, 'slides', N));
