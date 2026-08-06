const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';                    // 13.33 x 7.5 in

const INK   = '1F2937';   // structure + text
const MUT   = '6B7280';   // muted labels
const ACC   = 'B45309';   // accent text
const ACCL  = 'D97706';   // accent stroke
const ACCBG = 'FFFBEB';   // accent fill
const FRZBG = 'F9FAFB';
const FRZLN = '9CA3AF';

const s = pres.addSlide();
s.background = { color: 'FFFFFF' };

/* ---------- title ---------- */
s.addText('Where the three interventions attach', {
  x: 0.55, y: 0.32, w: 12.2, h: 0.55, margin: 0,
  fontFace: 'Calibri', fontSize: 32, bold: true, color: INK,
});
s.addText('Every hook attaches outside the frozen weights.  The fourth condition, '
        + 'Original policy, is this same loop with all three off.', {
  x: 0.55, y: 0.92, w: 12.2, h: 0.32, margin: 0,
  fontFace: 'Calibri', fontSize: 13, color: MUT,
});

/* ---------- helpers ---------- */
const box = (x, y, w, h, opts = {}) => ({
  x, y, w, h, shape: pres.ShapeType.roundRect, rectRadius: 0.04,
  fontFace: 'Calibri', align: 'center', valign: 'middle', margin: 0,
  color: INK, ...opts,
});
const arrow = (x, y, w, h, color = INK) => s.addShape(pres.ShapeType.line, {
  x, y, w, h, line: { color, width: 1.75, endArrowType: 'triangle' },
});
const note = (t, x, y, w, opts = {}) => s.addText(t, {
  x, y, w, h: 0.22, margin: 0, fontFace: 'Calibri', fontSize: 10,
  color: MUT, align: 'center', ...opts,
});

/* ---------- inputs ---------- */
s.addText([{ text: 'RGB frame', options: { fontSize: 13, breakLine: true } },
           { text: 'env camera', options: { fontSize: 10, color: MUT } }],
  box(0.55, 2.58, 1.55, 0.95, { fill: 'FFFFFF', line: { color: INK, width: 1.4 } }));

s.addText('Instruction',
  box(0.55, 4.00, 1.55, 0.65, { fontSize: 13, fill: 'FFFFFF', line: { color: INK, width: 1.4 } }));

/* ---------- HOOK A ---------- */
note('HOOK A  ·  the observation', 2.15, 1.92, 2.60, { color: ACC, bold: true, fontSize: 11 });
s.addShape(pres.ShapeType.roundRect, {
  x: 2.42, y: 2.20, w: 2.06, h: 1.70, rectRadius: 0.05,
  fill: ACCBG, line: { color: ACCL, width: 1.6, dashType: 'dash' },
});
s.addText('①  Fixed foveation', {
  x: 2.42, y: 2.30, w: 2.06, h: 0.28, margin: 0,
  fontFace: 'Calibri', fontSize: 13, bold: true, color: ACC, align: 'center',
});
s.addShape(pres.ShapeType.rect, { x: 3.00, y: 2.70, w: 0.92, h: 0.72,
  fill: 'E5E7EB', line: { color: FRZLN, width: 1 } });
s.addShape(pres.ShapeType.rect, { x: 3.28, y: 2.88, w: 0.36, h: 0.36,
  fill: 'FFFFFF', line: { color: ACCL, width: 1.4 } });
note('sharp centre, degraded edge', 2.15, 3.50, 2.60);

/* ---------- frozen backbone ---------- */
s.addShape(pres.ShapeType.roundRect, {
  x: 4.90, y: 2.10, w: 3.35, h: 3.55, rectRadius: 0.04,
  fill: FRZBG, line: { color: FRZLN, width: 1.4 },
});
s.addText('Frozen VLA backbone', {
  x: 4.90, y: 2.22, w: 3.35, h: 0.30, margin: 0,
  fontFace: 'Calibri', fontSize: 15, bold: true, color: INK, align: 'center',
});
note('no training, no added modules', 4.90, 2.53, 3.35);

s.addText('Vision encoder',
  box(5.15, 2.85, 2.85, 0.45, { fontSize: 12, fill: 'FFFFFF', line: { color: MUT, width: 1.2 } }));

s.addShape(pres.ShapeType.roundRect, {
  x: 5.15, y: 3.60, w: 2.85, h: 1.15, rectRadius: 0.03,
  fill: 'FFFFFF', line: { color: MUT, width: 1.2 },
});
s.addText('LLM decoder stack', {
  x: 5.27, y: 3.66, w: 1.7, h: 0.22, margin: 0,
  fontFace: 'Calibri', fontSize: 11, color: INK, align: 'left',
});

// six decoder layers; #2 and #4 bypassed
const LX = [5.32, 5.75, 6.18, 6.61, 7.04, 7.47];
const SKIP = [1, 3];
LX.forEach((x, i) => {
  const off = SKIP.includes(i);
  s.addShape(pres.ShapeType.rect, {
    x, y: 4.02, w: 0.30, h: 0.32,
    fill: off ? ACCBG : 'FFFFFF',
    line: { color: off ? ACCL : MUT, width: off ? 1.4 : 1.1, dashType: off ? 'dash' : 'solid' },
  });
});
SKIP.forEach(i => arrow(LX[i] - 0.09, 3.90, 0.48, 0, ACCL));
note('③  bypass the N most redundant layers', 4.90, 4.42, 3.35,
     { color: ACC, bold: true, fontSize: 11 });
note('HOOK C', 6.95, 4.72, 1.30, { color: ACC, bold: true, fontSize: 11, align: 'right' });

s.addText('Action decoder',
  box(5.15, 5.05, 2.85, 0.45, { fontSize: 12, fill: 'FFFFFF', line: { color: MUT, width: 1.2 } }));

/* ---------- HOOK B ---------- */
note('HOOK B  ·  the action', 8.58, 4.30, 2.60, { color: ACC, bold: true, fontSize: 11 });
s.addShape(pres.ShapeType.roundRect, {
  x: 8.85, y: 4.58, w: 2.06, h: 1.32, rectRadius: 0.05,
  fill: ACCBG, line: { color: ACCL, width: 1.6, dashType: 'dash' },
});
s.addText('②  Action repeat', {
  x: 8.85, y: 4.68, w: 2.06, h: 0.28, margin: 0,
  fontFace: 'Calibri', fontSize: 13, bold: true, color: ACC, align: 'center',
});
s.addText('a', box(9.06, 5.06, 0.40, 0.34, { fontSize: 12, fill: 'FFFFFF', line: { color: MUT, width: 1.1 } }));
arrow(9.53, 5.23, 0.20, 0);
s.addText('a', box(9.80, 5.06, 0.40, 0.34, { fontSize: 12, color: ACC, fill: 'FFFFFF', line: { color: ACCL, width: 1.2 } }));
s.addText('a', box(10.25, 5.06, 0.40, 0.34, { fontSize: 12, color: ACC, fill: 'FFFFFF', line: { color: ACCL, width: 1.2 } }));
note('hold it for N env steps', 8.68, 5.50, 2.40);

/* ---------- environment ---------- */
s.addText([{ text: 'Simulator', options: { fontSize: 13, breakLine: true } },
           { text: 'env.step() on', options: { fontSize: 9, color: MUT, breakLine: true } },
           { text: 'unmodified scene', options: { fontSize: 9, color: MUT } }],
  box(11.25, 4.71, 1.55, 1.06, { fill: 'FFFFFF', line: { color: INK, width: 1.4 } }));

/* ---------- flow ---------- */
arrow(2.10, 3.05, 0.28, 0);                       // obs -> hook A
arrow(4.48, 3.05, 0.38, 0);                       // hook A -> backbone
arrow(2.10, 4.32, 2.76, 0);                       // instruction -> LLM
arrow(6.57, 3.30, 0, 0.26);                       // vision -> LLM
arrow(6.57, 4.75, 0, 0.26);                       // LLM -> action decoder
arrow(8.00, 5.27, 0.81, 0);                       // action decoder -> hook B
note('1 action', 7.90, 5.02, 1.00);
arrow(10.91, 5.27, 0.30, 0);                      // hook B -> env
note('N steps', 10.55, 5.02, 1.00);

/* feedback loop */
const FB = { color: MUT, width: 1.4, dashType: 'dash' };
s.addShape(pres.ShapeType.line, { x: 12.02, y: 5.77, w: 0, h: 0.63, line: FB });
s.addShape(pres.ShapeType.line, { x: 1.32, y: 6.40, w: 10.70, h: 0, line: FB });
s.addShape(pres.ShapeType.line, { x: 1.32, y: 6.40, w: 0, h: -2.87,
  line: { ...FB, endArrowType: 'triangle' } });
note('next observation', 6.10, 6.44, 1.60);

s.addNotes('Methods figure for the four-condition grid. Three of the four attach somewhere -- '
         + 'fixed foveation (hook A), action repeat (hook B), fixed depth pruning (hook C); '
         + 'the fourth, original policy, is the same loop with all three off. '
         + 'All shapes are native PowerPoint objects and can be edited directly.');

pres.writeFile({ fileName: '/home/user/BiVLA/experiments/figures/hook_points.pptx' })
    .then(f => console.log('wrote', f));
