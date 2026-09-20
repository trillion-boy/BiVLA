// Video-wall slides and paired clips for the trick slides (mentor's plan).
const path = require('path');
const fs = require('fs');
const L = require('./lib');
const { C, FONT, HFONT, title, caption, notes } = L;
const M = JSON.parse(fs.readFileSync(path.join(__dirname, 'clips', 'manifest.json'), 'utf8'));
const CL = p => path.join(__dirname, p);
const b64 = p => 'image/png;base64,' + fs.readFileSync(p).toString('base64');

const LABEL = {
  original: 'Original', fixed_foveation_keep20: 'Foveation 20%', fixed_foveation_keep50: 'Foveation 50%',
  action_repeat2: 'Action repeat 2', action_repeat4: 'Action repeat 4',
  depth_pruning1: 'Depth pruning 1', depth_pruning2: 'Depth pruning 2', depth_pruning4: 'Depth pruning 4',
  guarded_reuse_strict: 'Reuse strict', guarded_reuse_moderate: 'Reuse moderate', guarded_reuse_aggressive: 'Reuse aggressive',
  temporal_fusion_motion_entropy: 'Fusion motion', temporal_fusion_task_aware: 'Fusion task-aware', temporal_fusion_conservative_adaptive: 'Fusion conserv.',
};
const ORDER = ['original', 'fixed_foveation_keep20', 'fixed_foveation_keep50', 'action_repeat2', 'action_repeat4', 'depth_pruning1', 'depth_pruning2', 'depth_pruning4',
  'guarded_reuse_strict', 'guarded_reuse_moderate', 'guarded_reuse_aggressive', 'temporal_fusion_motion_entropy', 'temporal_fusion_task_aware', 'temporal_fusion_conservative_adaptive'];
const COL = { original: '222222', fixed: '0072B2', action: 'D55E00', depth: '009E73', guarded: 'CC79A7', temporal: '8C650F' };
const colOf = cfg => COL[cfg.split('_')[0]] || C.ink;
const TASK = {
  widowx_carrot_on_plate: 'Carrot on plate', widowx_put_eggplant_in_basket: 'Eggplant in basket', widowx_spoon_on_towel: 'Spoon on towel', widowx_stack_cube: 'Stack cube',
  google_robot_close_drawer: 'Close drawer', google_robot_move_near: 'Move near', google_robot_pick_coke_can: 'Pick coke can', google_robot_open_drawer: 'Open drawer',
};

function video(s, clip, x, y, w, h) {
  s.addMedia({ type: 'video', path: CL(clip.out), cover: b64(CL(clip.out.replace('.mp4', '.png'))), x, y, w, h });
}

// One wall slide: every configuration of one backbone on one task, same episode, playing together.
function wall(pres, S, wallSpec, n) {
  const secs = Math.ceil(wallSpec.maxdur);
  const s = S(secs);
  const isW = wallSpec.env === 'widowx';
  const bb = isW ? 'CogACT, WidowX' : 'OpenVLA, Fractal';
  title(s, `${bb}: ${TASK[wallSpec.task]}`, { size: 24 });
  caption(s, `All ${wallSpec.tiles.length} configurations on the same episode (same initial state), ${wallSpec.speed}× speed. Green border = success, red = failure.`, 0.5, 0.95, 9.0, { size: 10.5, h: 0.28 });
  const tiles = ORDER.filter(c => wallSpec.tiles.some(t => t.cfg === c)).map(c => wallSpec.tiles.find(t => t.cfg === c));
  let cols, tw, th, x0, y0, gx, gy, lh;
  if (isW) { cols = 5; tw = 1.56; th = 1.17; gx = 0.3; gy = 0.04; lh = 0.2; x0 = 0.5; y0 = 1.3; }
  else { cols = 6; tw = 1.22; th = 1.57; gx = 0.336; gy = 0.1; lh = 0.2; x0 = 0.5; y0 = 1.35; }
  tiles.forEach((t, i) => {
    const r = Math.floor(i / cols), c = i % cols;
    const x = x0 + c * (tw + gx), y = y0 + r * (th + lh + gy);
    s.addText(LABEL[t.cfg], { x, y, w: tw, h: lh, fontFace: FONT, fontSize: 9, bold: true, color: colOf(t.cfg), margin: 0, isTextBox: true, valign: 'bottom' });
    video(s, t, x, y + lh, tw, th);
    s.addShape('rect', { x, y: y + lh, w: tw, h: th, fill: { type: 'none' }, line: { color: t.success ? '2E8B57' : 'B03A2E', width: 1.5 } });
  });
  const ok = tiles.filter(t => t.success).length;
  const first = wallSpec === M.walls[0], firstF = wallSpec === M.walls[3];
  notes(s, n, secs, first
    ? `Now every configuration on one episode. CogACT on WidowX, ${ok} of ${tiles.length} succeed.`
    : firstF ? `OpenVLA on Fractal, same episode for every configuration. ${ok} of ${tiles.length} succeed.`
    : `${TASK[wallSpec.task]}: ${ok} of ${tiles.length} succeed.`,
    `Video wall, ${tiles.length} clips at ${wallSpec.speed}x, longest ${wallSpec.maxdur} s; the slide auto-advances after ${secs} s. Clips auto-play on slide entry.`);
  return secs;
}

// Paired clip (original vs trick) for a trick slide, in the 4.3 x 3.05 in area at x=5.2, y=1.05
function pair(s, key) {
  const p = M.pairs[key];
  const w = 2.08, h = 1.56, y = 1.12;
  [[p.orig, 'Original', 5.2], [p.trick, LABEL[p.cfg], 5.2 + w + 0.14]].forEach(([clip, lab, x]) => {
    s.addText(lab, { x, y: y - 0.22, w, h: 0.2, fontFace: FONT, fontSize: 9.5, bold: true, color: C.ink, margin: 0, isTextBox: true, valign: 'bottom' });
    video(s, clip, x, y, w, h);
    s.addShape('rect', { x, y, w, h, fill: { type: 'none' }, line: { color: clip.success ? '2E8B57' : 'B03A2E', width: 1.5 } });
  });
  caption(s, `CogACT on WidowX, ${TASK[p.task]}, same episode, 2× speed. Green border = success, red = failure.`, 5.2, y + h + 0.08, 4.3, { size: 10, h: 0.45 });
}

module.exports = { wall, pair, M };
