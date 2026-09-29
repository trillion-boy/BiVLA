(() => {
  const order = [
    'fixed_foveation_keep50', 'fixed_foveation_keep20', 'action_repeat2', 'action_repeat4',
    'depth_pruning1', 'depth_pruning2', 'depth_pruning4', 'guarded_reuse_strict',
    'guarded_reuse_moderate', 'guarded_reuse_aggressive', 'temporal_fusion_conservative_adaptive',
    'temporal_fusion_motion_entropy', 'temporal_fusion_task_aware'
  ];
  const labels = {
    fixed_foveation_keep50: 'Foveation: keep 50%', fixed_foveation_keep20: 'Foveation: keep 20%',
    action_repeat2: 'Action repeat: 2', action_repeat4: 'Action repeat: 4',
    depth_pruning1: 'Depth pruning: 1 layer', depth_pruning2: 'Depth pruning: 2 layers',
    depth_pruning4: 'Depth pruning: 4 layers', guarded_reuse_strict: 'Reuse: strict',
    guarded_reuse_moderate: 'Reuse: moderate', guarded_reuse_aggressive: 'Reuse: aggressive',
    temporal_fusion_conservative_adaptive: 'Fusion: conservative',
    temporal_fusion_motion_entropy: 'Fusion: motion/entropy', temporal_fusion_task_aware: 'Fusion: task-aware'
  };
  const colors = { Foveation: '#1676a3', 'Action repeat': '#df6b13', 'Depth pruning': '#15966f', 'Guarded reuse': '#c86e9d', 'Temporal fusion': '#987112' };
  const family = {
    fixed_foveation_keep50: 'Foveation', fixed_foveation_keep20: 'Foveation',
    action_repeat2: 'Action repeat', action_repeat4: 'Action repeat',
    depth_pruning1: 'Depth pruning', depth_pruning2: 'Depth pruning', depth_pruning4: 'Depth pruning',
    guarded_reuse_strict: 'Guarded reuse', guarded_reuse_moderate: 'Guarded reuse', guarded_reuse_aggressive: 'Guarded reuse',
    temporal_fusion_conservative_adaptive: 'Temporal fusion', temporal_fusion_motion_entropy: 'Temporal fusion', temporal_fusion_task_aware: 'Temporal fusion'
  };
  const svgEl = (name, attrs = {}, content = '') => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (content) node.textContent = content;
    return node;
  };
  const familyColor = config => colors[family[config]] || '#1f5f8b';

  Promise.resolve().then(() => {
    const rows = JSON.parse(document.getElementById('chart-data').textContent);
    const numeric = ['episodes', 'successes', 'success_pct', 'delta_success_vs_original', 'p_mcnemar'];
    rows.forEach(row => numeric.forEach(key => { row[key] = row[key] === '' ? null : Number(row[key]); }));
    const consistencyWrap = document.querySelector('#consistency-chart');
    const consistencySVG = consistencyWrap.querySelector('svg');
    const changeWrap = document.querySelector('#change-chart');
    const changeSVG = changeWrap.querySelector('svg');
    const envSelect = document.querySelector('#chart-environment');
    const backboneSelect = document.querySelector('#chart-backbone');
    const status = document.querySelector('#chart-status');

    const groupedPairs = new Map();
    rows.filter(r => r.configuration === 'original' || order.includes(r.configuration)).forEach(r => {
      const pooledEnv = r.env.startsWith('LIBERO') ? 'LIBERO' : r.env;
      const key = [r.backbone, pooledEnv, r.configuration].join('|');
      if (!groupedPairs.has(key)) groupedPairs.set(key, { backbone: r.backbone, env: pooledEnv, configuration: r.configuration, episodes: 0, successes: 0, p: null });
      const g = groupedPairs.get(key); g.episodes += r.episodes; g.successes += r.successes;
      if (pooledEnv !== 'LIBERO') g.p = r.p_mcnemar;
    });
    const pairRows = [...groupedPairs.values()].map(r => ({ ...r, success_pct: 100 * r.successes / r.episodes }));
    const byPairConfig = new Map(pairRows.map(r => [[r.backbone, r.env, r.configuration].join('|'), r]));

    const renderConsistency = () => {
      consistencySVG.replaceChildren();
      consistencySVG.setAttribute('viewBox', '0 0 920 490');
      const x0 = 230, x1 = 850, top = 22, rowH = 31, scale = (x1 - x0) / 13;
      for (let tick = 0; tick <= 13; tick += 2) {
        const x = x0 + tick * scale;
        consistencySVG.append(svgEl('line', { x1: x, y1: top - 7, x2: x, y2: top + rowH * order.length - 4, class: tick === 0 ? 'axis' : 'grid' }));
        consistencySVG.append(svgEl('text', { x, y: top + rowH * order.length + 17, 'text-anchor': 'middle' }, String(tick)));
      }
      order.forEach((config, index) => {
        const y = top + index * rowH;
        consistencySVG.append(svgEl('text', { x: x0 - 12, y: y + 15, 'text-anchor': 'end' }, labels[config]));
        let low = 0, same = 0, high = 0;
        const pairs = pairRows.filter(r => r.configuration === config);
        pairs.forEach(r => {
          const original = byPairConfig.get([r.backbone, r.env, 'original'].join('|'));
          const delta = r.success_pct - original.success_pct;
          if (delta < -1e-9) low++; else if (delta > 1e-9) high++; else same++;
        });
        let x = x0;
        [[low, '#c85b52', 'Lower success'], [same, '#d7dadd', 'Unchanged success'], [high, '#347f9e', 'Higher success']].forEach(([count, fill, name]) => {
          if (count) {
            const bar = svgEl('rect', { x, y: y + 2, width: count * scale, height: 22, rx: 2, fill, class: 'bar', tabindex: 0, 'aria-label': `${labels[config]}: ${count} of 13 pairs, ${name.toLowerCase()}` });
            bar.append(svgEl('title', {}, `${labels[config]} — ${name}: ${count} of 13 backbone/environment pairs`));
            consistencySVG.append(bar);
            if (count >= 1) consistencySVG.append(svgEl('text', { x: x + count * scale / 2, y: y + 17, 'text-anchor': 'middle', 'pointer-events': 'none' }, String(count)));
            x += count * scale;
          }
        });
      });
      consistencySVG.append(svgEl('text', { x: (x0 + x1) / 2, y: 480, 'text-anchor': 'middle' }, 'Backbone and environment pairs (13 total)'));
    };

    const renderChange = () => {
      const env = envSelect.value, backbone = backboneSelect.value;
      const plotted = order.map(config => byPairConfig.get([backbone, env, config].join('|'))).filter(Boolean);
      const original = byPairConfig.get([backbone, env, 'original'].join('|'));
      if (!plotted.length || !original) { status.textContent = 'No matching results are available.'; return; }
      const changes = plotted.map(r => r.success_pct - original.success_pct);
      const min = Math.min(-2, Math.floor(Math.min(...changes) / 5) * 5);
      const max = Math.max(2, Math.ceil(Math.max(...changes) / 5) * 5);
      const viewW = 920, left = 245, right = 895, top = 24, rowH = 31, chartBottom = top + rowH * plotted.length;
      const x = value => left + (value - min) / (max - min) * (right - left);
      changeSVG.replaceChildren();
      changeSVG.setAttribute('viewBox', `0 0 ${viewW} ${chartBottom + 52}`);
      const step = (max - min) > 45 ? 10 : (max - min) > 25 ? 5 : (max - min) > 12 ? 2 : 1;
      const firstTick = Math.ceil(min / step) * step;
      for (let tick = firstTick; tick <= max; tick += step) {
        const tx = x(tick);
        changeSVG.append(svgEl('line', { x1: tx, y1: top - 8, x2: tx, y2: chartBottom - 4, class: tick === 0 ? 'axis' : 'grid' }));
        changeSVG.append(svgEl('text', { x: tx, y: chartBottom + 16, 'text-anchor': 'middle' }, `${tick > 0 ? '+' : ''}${tick}`));
      }
      plotted.forEach((r, i) => {
        const y = top + i * rowH;
        const delta = r.success_pct - original.success_pct;
        const xA = x(Math.min(0, delta)), xB = x(Math.max(0, delta));
        changeSVG.append(svgEl('text', { x: left - 12, y: y + 15, 'text-anchor': 'end' }, labels[r.configuration]));
        const bar = svgEl('rect', { x: xA, y: y + 3, width: Math.max(1.5, xB - xA), height: 21, rx: 2, fill: familyColor(r.configuration), class: 'bar', tabindex: 0,
          'aria-label': `${labels[r.configuration]}: ${delta >= 0 ? '+' : ''}${delta.toFixed(1)} percentage points; ${r.successes} of ${r.episodes} successes` });
        const pText = env === 'Fractal' && r.p !== null ? `; exact McNemar p = ${r.p < 0.001 ? '< 0.001' : r.p.toFixed(3)}` : '';
        bar.append(svgEl('title', {}, `${labels[r.configuration]}: ${delta >= 0 ? '+' : ''}${delta.toFixed(1)} percentage points (${r.successes}/${r.episodes} vs ${original.successes}/${original.episodes} original successes)${pText}`));
        changeSVG.append(bar);
      });
      changeSVG.append(svgEl('text', { x: (left + right) / 2, y: chartBottom + 43, 'text-anchor': 'middle' }, 'Success change from original (percentage points)'));
      const suiteNote = env === 'LIBERO' ? `LIBERO pooled: ${backbone}, ${original.episodes} episodes across four suites.` : `${env}: ${backbone}, ${original.episodes} episodes.`;
      status.textContent = `${suiteNote} Hover or focus a bar for counts and the paired p-value.`;
    };

    const updateBackbones = () => {
      const previous = backboneSelect.value;
      const choices = [...new Set(pairRows.filter(r => r.env === envSelect.value).map(r => r.backbone))].sort();
      backboneSelect.replaceChildren(...choices.map(name => { const opt = document.createElement('option'); opt.value = name; opt.textContent = name; return opt; }));
      if (choices.includes(previous)) backboneSelect.value = previous;
      renderChange();
    };
    renderConsistency();
    updateBackbones();
    envSelect.addEventListener('change', updateBackbones);
    backboneSelect.addEventListener('change', renderChange);
    consistencyWrap.hidden = false; changeWrap.hidden = false;
    document.querySelectorAll('.chart-fallback').forEach(img => { img.hidden = true; });
  }).catch(error => {
    const status = document.querySelector('#chart-status');
    if (status) status.textContent = `${error.message} Showing the static chart preview.`;
  });
})();
