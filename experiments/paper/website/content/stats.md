**Per-cell p-values.** Every setting's exact two-sided McNemar p-value, computed on the episodes whose outcome flipped between the original and the trick, is in the tables of 2.1. Over the 110 table cells (the best setting per trick), 22 lower success at p < 0.05 and 4 raise it: CogACT WidowX depth pruning 2 layers (+9.5, p = 0.009), OpenVLA Fractal task-aware fusion (+6.8, p = 0.012), CogACT WidowX motion-entropy fusion (+8.0, p = 0.020) and UniVLA LIBERO Goal foveation 50% (+7.0, p = 0.039). Over all 286 settings, 77 lower success and 6 raise it at p < 0.05.

**Multiplicity.** The paper reads the p-values as descriptive because each table cell is the best of two or three settings. A Benjamini-Hochberg correction at q = 0.05 over the 110 table cells keeps 16 of the 22 drops (cut-off p = 0.0061) and none of the 4 gains. The sum of the exact test sizes at the observed flip counts gives 3.1 chance passes over the 110 cells and 8.0 over the 286 settings, 4.0 per direction, so the 6 gains over all settings are about what chance produces. Guarded reuse changes success significantly in none of its 66 settings.

<details markdown="1"><summary>The 22 significant table drops and whether they pass the correction</summary>

| Backbone | Environment | Setting | Change | p | BH q = 0.05 |
|---|---|---|---|---|---|
| CogACT | WidowX | action repeat 2 | -38.0 | < 0.001 | pass |
| CronusVLA | WidowX | action repeat 2 | -30.5 | < 0.001 | pass |
| MiniVLA | WidowX | depth pruning 1 | -17.5 | < 0.001 | pass |
| UniVLA | WidowX | action repeat 2 | -75.0 | < 0.001 | pass |
| UniVLA | LIBERO Long | action repeat 2 | -59.0 | < 0.001 | pass |
| UniVLA | LIBERO Goal | action repeat 2 | -58.0 | < 0.001 | pass |
| UniVLA | LIBERO Object | action repeat 2 | -75.0 | < 0.001 | pass |
| UniVLA | LIBERO Spatial | action repeat 2 | -68.0 | < 0.001 | pass |
| SmolVLA | LIBERO Long | depth pruning 1 | -33.0 | < 0.001 | pass |
| SmolVLA | LIBERO Object | depth pruning 1 | -21.0 | < 0.001 | pass |
| SmolVLA | LIBERO Long | foveation 50% | -21.0 | < 0.001 | pass |
| OpenVLA | WidowX | action repeat 2 | -15.5 | < 0.001 | pass |
| OpenVLA | Fractal | foveation 50% | -11.2 | 0.001 | pass |
| SmolVLA | LIBERO Goal | depth pruning 1 | -18.0 | 0.001 | pass |
| SmolVLA | LIBERO Object | foveation 50% | -13.0 | 0.002 | pass |
| OpenVLA | WidowX | foveation 50% | -13.5 | 0.006 | pass |
| UniVLA | WidowX | foveation 20% | -8.5 | 0.010 | fail |
| OpenVLA | LIBERO Spatial | depth pruning 1 | -16.0 | 0.011 | fail |
| SmolVLA | LIBERO Goal | foveation 50% | -13.0 | 0.015 | fail |
| SpatialVLA | WidowX | depth pruning 1 | -6.5 | 0.024 | fail |
| SmolVLA | LIBERO Spatial | depth pruning 1 | -14.0 | 0.034 | fail |
| OpenVLA | LIBERO Long | foveation 50% | -12.0 | 0.036 | fail |

</details>

**Run-to-run noise.** On a guarded-reuse episode where the gate never fired, the original policy was called at every step, so the outcome should match the original run of the same seed. It does on SpatialVLA, UniVLA, OpenVLA Fractal and OpenVLA LIBERO (0.0 to 0.7 percent of such episodes differ). It does not on four WidowX pairs: the outcome differs on 16.5 percent of zero-fire episodes for CogACT, 13.4 for OpenVLA, 9.3 for CronusVLA and 9.1 for MiniVLA, and on 1.8 to 4.6 percent on CogACT and CronusVLA Fractal. On those pairs a cell can move by a few points with no trick at all, which is the scale of several small gains in Table I (CogACT WidowX foveation +2.5 at p = 0.60, guarded reuse +3.0 at p = 0.36, CronusVLA WidowX foveation +2.5 at p = 0.44), and only the paired test is read. The SmolVLA reuse cells, run under the re-implemented evaluator, differ from the original on 10.5 to 30.5 percent of zero-fire episodes.

| Backbone | WidowX | Fractal | LIBERO (pooled) |
|---|---|---|---|
| CogACT | 16.5 % (522 zero-fire episodes) | 1.8 % (716) | |
| OpenVLA | 13.4 % (409) | 0.0 % (612) | 0.0 % (144) |
| CronusVLA | 9.3 % (538) | 4.6 % (609) | |
| MiniVLA | 9.1 % (438) | | |
| SpatialVLA | 0.5 % (572) | 0.0 % (604) | |
| UniVLA | 0.3 % (598) | | 0.2 % (1,186) |
| SmolVLA | | | 20.3 % (760) |

**Cells that are the original policy under another name.** A zero change can mean the trick did not engage. The ten UniVLA guarded-reuse cells on Goal, Object, Spatial and WidowX strict fired no gate and differ from their original row by at most 1.6 percent in latency, which is the latency floor of a repeated run. Conservative-adaptive fusion reused no patch on SpatialVLA or UniVLA, so the Table I UniVLA WidowX fusion cell (87.50, +0.00) and the Table II UniVLA Spatial fusion cell (95.00, +0.00) are the original run. CogACT's task-aware fusion is its motion-entropy run, and CronusVLA's three fusion settings are one run.
