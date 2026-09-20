// Shared design system for the ICRA video storyboard deck (16:9, 10 x 5.625 in)
const path = require('path');
const A = p => path.join(__dirname, 'assets', p);

const C = {
  ink: '1F2933',      // body text
  muted: '5F6B7A',    // captions
  dark: '16213A',     // dark slide background
  darkText: 'F5F7FA',
  card: 'F3F5F7',     // light card tint
  line: 'D9DEE3',
  teal: '1B7F79',     // primary accent
  tealSoft: 'DDEFEC',
  orange: 'C8641E',   // warning / negative
  orangeSoft: 'F9E6D8',
  blue: '2F6DB5',
  blueSoft: 'E1EAF7',
  green: '2E8B57',
  greenSoft: 'DFF1E6',
  red: 'B03A2E',
  redSoft: 'F6E0DD',
  white: 'FFFFFF',
};
const FONT = 'Arial';
const HFONT = 'Arial';

function base(slide, dark = false) {
  slide.background = { color: dark ? C.dark : C.white };
}

function title(slide, text, opts = {}) {
  const dark = !!opts.dark;
  slide.addText(text, {
    x: 0.5, y: 0.32, w: 9.0, h: 0.7, fontFace: HFONT, fontSize: opts.size || 26, bold: true,
    color: dark ? C.darkText : C.ink, margin: 0, isTextBox: true, valign: 'middle',
  });
}

function timeTag(slide, n, secs, dark = false) {
  slide.addText(`${n}  ·  ${secs}s`, {
    x: 8.4, y: 0.08, w: 1.4, h: 0.25, fontFace: FONT, fontSize: 9, color: dark ? 'A9B4C2' : C.muted,
    align: 'right', margin: 0, isTextBox: true,
  });
}

function caption(slide, text, x, y, w, opts = {}) {
  slide.addText(text, {
    x, y, w, h: opts.h || 0.3, fontFace: FONT, fontSize: opts.size || 11, color: opts.color || C.muted,
    italic: !!opts.italic, align: opts.align || 'left', margin: 0, isTextBox: true, valign: 'top',
  });
}

function body(slide, lines, x, y, w, h, opts = {}) {
  const size = opts.size || 16;
  const items = lines.map((t, i) => ({
    text: t, options: { bullet: opts.bullet !== false ? { indent: 14 } : false, breakLine: i < lines.length - 1, paraSpaceAfter: opts.gap || 6 },
  }));
  slide.addText(items, {
    x, y, w, h, fontFace: FONT, fontSize: size, color: opts.color || C.ink, margin: 0, isTextBox: true, valign: 'top',
  });
}

function card(slide, x, y, w, h, opts = {}) {
  slide.addShape('roundRect', {
    x, y, w, h, fill: { color: opts.fill || C.card }, line: { color: opts.line || C.line, width: opts.lineW ?? 0.75 },
    rectRadius: opts.r ?? 0.12,
  });
}

// Small circle badge with a number or short label
function badge(slide, label, x, y, opts = {}) {
  const d = opts.d || 0.42;
  slide.addShape('ellipse', { x, y, w: d, h: d, fill: { color: opts.fill || C.teal }, line: { color: opts.fill || C.teal, width: 0 } });
  slide.addText(label, {
    x, y, w: d, h: d, fontFace: HFONT, fontSize: opts.size || 13, bold: true, color: C.white,
    align: 'center', valign: 'middle', margin: 0, isTextBox: true,
  });
}

// Placeholder box for a rollout clip the team will record
function clipPlaceholder(slide, x, y, w, h, label, sub) {
  slide.addShape('roundRect', { x, y, w, h, fill: { color: '2A2F3A' }, line: { color: '8A94A6', width: 1, dashType: 'dash' }, rectRadius: 0.08 });
  if (h < 1.3) {
    slide.addText('▶  ' + label, { x: x + 0.1, y, w: w - 0.2, h, fontFace: FONT, fontSize: 12, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    return;
  }
  slide.addShape('ellipse', { x: x + w / 2 - 0.3, y: y + h / 2 - 0.42, w: 0.6, h: 0.6, fill: { color: 'FFFFFF' }, line: { color: 'FFFFFF', width: 0 } });
  slide.addShape('triangle', { x: x + w / 2 - 0.09, y: y + h / 2 - 0.29, w: 0.24, h: 0.34, rotate: 90, fill: { color: '2A2F3A' }, line: { color: '2A2F3A', width: 0 } });
  slide.addText(label, { x: x + 0.1, y: y + h / 2 + 0.25, w: w - 0.2, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: 'FFFFFF', align: 'center', margin: 0, isTextBox: true });
  if (sub) slide.addText(sub, { x: x + 0.1, y: y + h / 2 + 0.52, w: w - 0.2, h: 0.4, fontFace: FONT, fontSize: 9.5, color: 'C9D1DC', align: 'center', margin: 0, isTextBox: true, valign: 'top' });
}

function notes(slide, n, secs, narration, visual) {
  const words = narration.trim().split(/\s+/).length;
  slide.addNotes(`SLIDE ${n}  |  ${secs} s  |  ${words} words (${(words / secs).toFixed(1)} w/s)\n\nNARRATION:\n${narration}\n\nVISUAL / PRODUCTION NOTE:\n${visual}`);
  if (process.env.SUBS) {
    slide.addShape('rect', { x: 0, y: 5.13, w: 10, h: 0.495, fill: { color: C.dark }, line: { color: C.dark, width: 0 } });
    slide.addText(narration, { x: 0.35, y: 5.13, w: 9.3, h: 0.495, fontFace: FONT, fontSize: 10.5, color: C.darkText, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  }
}

// Big stat callout: number on top, label under
function stat(slide, value, label, x, y, w, opts = {}) {
  slide.addText(value, { x, y, w, h: 0.75, fontFace: HFONT, fontSize: opts.size || 40, bold: true, color: opts.color || C.teal, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  slide.addText(label, { x, y: y + 0.75, w, h: 0.5, fontFace: FONT, fontSize: opts.lsize || 12, color: opts.lcolor || C.muted, align: 'center', valign: 'top', margin: 0, isTextBox: true });
}

module.exports = { A, C, FONT, HFONT, base, title, timeTag, caption, body, card, badge, clipPlaceholder, notes, stat };
