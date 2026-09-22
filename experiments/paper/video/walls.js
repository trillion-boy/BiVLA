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
  libero_spatial__task_3: 'Bowl on cookie box to plate', libero_object__task_9: 'Orange juice into basket', libero_goal__task_8: 'Bowl on plate', libero_10__task_1: 'Cream cheese and butter into basket',
};
const SUITE = { libero_spatial: 'LIBERO Spatial', libero_object: 'LIBERO Object', libero_goal: 'LIBERO Goal', libero_10: 'LIBERO Long' };

function video(s, clip, x, y, w, h) {
  s.addMedia({ type: 'video', path: CL(clip.out), cover: b64(CL(clip.out.replace('.mp4', '.png'))), x, y, w, h });
}

// One wall slide: every configuration of one backbone on one task, same episode, playing together.
function wall(pres, S, wallSpec, n) {
  const secs = wallSpec.secs || Math.ceil(wallSpec.maxdur);
  const s = S(secs);
  const isW = wallSpec.env === 'widowx', isL = wallSpec.env === 'libero';
  const bb = isW ? 'CogACT, WidowX' : isL ? `UniVLA, ${SUITE[wallSpec.task.split('__')[0]]}` : 'OpenVLA, Fractal';
  title(s, `${bb}: ${TASK[wallSpec.task]}`, { size: isL ? 22 : 24 });
  const spd = `${wallSpec.speed}× speed`;
  caption(s, isL ? `All 14 configurations on the same episode, ${spd}, runs cut at the slide end. Green border = success, red = failure.`
    : `All 14 configurations on the same episode (same initial state), ${spd}. Green border = success, red = failure.`, 0.5, 0.95, 9.0, { size: 10.5, h: 0.28 });
  const tiles = ORDER.filter(c => wallSpec.tiles.some(t => t.cfg === c)).map(c => wallSpec.tiles.find(t => t.cfg === c));
  if (tiles.length !== 14) throw new Error(`wall ${wallSpec.env} ${wallSpec.task} has ${tiles.length} tiles`);
  const SUB = !!process.env.SUBS; // subtitle band occupies y >= 5.13, so every wall must end above 5.05
  let cols, tw, th, x0, y0, gx, gy, lh;
  if (isW) { cols = 5; lh = 0.2; gy = 0.04; x0 = 0.5; y0 = 1.3; th = SUB ? 1.01 : 1.17; tw = +(th * 4 / 3).toFixed(3); gx = +((9.0 - cols * tw) / (cols - 1)).toFixed(3); }
  else if (isL) { cols = 7; tw = 1.2; th = 1.415; gx = 0.1; gy = 0.1; lh = 0.2; x0 = 0.5; y0 = 1.45; }
  else { cols = 7; tw = 1.2; th = 1.543; gx = 0.1; gy = 0.1; lh = 0.2; x0 = 0.5; y0 = 1.35; }
  const rows = Math.ceil(tiles.length / cols), yEnd = y0 + rows * (lh + th + gy) - gy;
  if (yEnd > (SUB ? 5.05 : 5.5)) throw new Error(`wall ${wallSpec.task} ends at ${yEnd.toFixed(2)} in`);
  tiles.forEach((t, i) => {
    const r = Math.floor(i / cols), c = i % cols;
    const x = x0 + c * (tw + gx), y = y0 + r * (th + lh + gy);
    s.addText(LABEL[t.cfg], { x, y, w: tw, h: lh, fontFace: FONT, fontSize: 9, bold: true, color: colOf(t.cfg), margin: 0, isTextBox: true, valign: 'bottom' });
    video(s, t, x, y + lh, tw, th);
    s.addShape('rect', { x, y: y + lh, w: tw, h: th, fill: { type: 'none' }, line: { color: t.success ? '2E8B57' : 'B03A2E', width: 1.5 } });
  });
  const ok = tiles.filter(t => t.success).length;
  const firstOfEnv = M.walls.find(w => w.env === wallSpec.env) === wallSpec;
  const name = isL ? SUITE[wallSpec.task.split('__')[0]].replace('LIBERO ', '') : TASK[wallSpec.task];
  const origOk = tiles.find(t => t.cfg === 'original').success;
  const fails = tiles.filter(t => !t.success).map(t => t.cfg);
  const fam = { fixed: 'foveation', action: 'action repeat', depth: 'depth pruning', guarded: 'guarded reuse', temporal: 'temporal fusion' };
  const oneFamily = fails.length === 2 && fails.every(c => c.split('_')[0] === fails[0].split('_')[0]) ? fam[fails[0].split('_')[0]] : null;
  // Every sentence says "one episode" and "configurations": the counts are outcomes on this episode, not success rates.
  const count = `${ok} of 14 configurations succeed`;
  const lastWall = wallSpec === M.walls[M.walls.length - 1];
  const tail = oneFamily && (firstOfEnv || lastWall) ? `, the two ${oneFamily} runs fail` : '';
  const narr = firstOfEnv
    ? (isW ? `Now every configuration on one episode. CogACT on WidowX, eggplant: ${count}${tail}.`
      : isL ? `UniVLA on LIBERO, one episode per suite. Spatial: ${count}${tail}.`
      : `OpenVLA on Fractal, again one episode per slide. Move near: ${count}${tail}.`)
    : !origOk ? `${name}, one episode: the original fails, ${count}.`
    : `${name}, one episode: ${count}${tail}.`;
  notes(s, n, secs, narr,
    `Video wall, ${tiles.length} clips at ${wallSpec.speed}x${wallSpec.cut ? ', clips cut at the slide length' : `, longest ${wallSpec.maxdur} s`}; the slide auto-advances after ${secs} s. Clips auto-play on slide entry.`);
  return secs;
}

// Paired clip (original vs trick) for a trick slide, in the 4.3 x 3.05 in area at x=5.2, y=1.05
function pair(s, key) {
  const p = M.pairs[key];
  const portrait = p.env === 'fractal';            // Fractal clips are 224 x 288, WidowX 640 x 480
  const h = portrait ? 1.9 : 1.56, w = portrait ? +(h * 224 / 288).toFixed(2) : 2.08, y = 1.12, gap = 0.14;
  const x0 = 5.2 + (4.3 - (2 * w + gap)) / 2;      // centred in the 4.3 in column
  [[p.orig, 'Original', x0], [p.trick, LABEL[p.cfg], x0 + w + gap]].forEach(([clip, lab, x]) => {
    s.addText(lab, { x, y: y - 0.22, w, h: 0.2, fontFace: FONT, fontSize: 9.5, bold: true, color: C.ink, margin: 0, isTextBox: true, valign: 'bottom' });
    video(s, clip, x, y, w, h);
    s.addShape('rect', { x, y, w, h, fill: { type: 'none' }, line: { color: clip.success ? '2E8B57' : 'B03A2E', width: 1.5 } });
  });
  const bb = p.env === 'widowx' ? 'CogACT on WidowX' : 'OpenVLA on Fractal';
  caption(s, `${bb}, ${TASK[p.task]}, same episode, ${p.speed}× speed. Green border = success, red = failure.`, 5.2, y + h + 0.08, 4.3, { size: 10, h: 0.45 });
}

module.exports = { wall, pair, M };
