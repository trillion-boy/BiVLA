# Per-cell paired test results

Every trick setting is compared with the original policy on the same episodes (same task instances and seeds). `n` is the number of paired episodes, `fail->success` the episodes the trick turned from failure into success, `success->fail` the reverse, `delta` the change in success rate in points, and `p` the exact two-sided McNemar test on the flipped episodes. `table` marks the setting shown in Table I or II (highest success, then fewer average steps). Each table cell is the best of two or three settings, so its p-value is post-selection and should be read as descriptive. Over all 286 settings, about eight would pass p<0.05 by chance at the observed flip counts.

SmolVLA guarded reuse and temporal fusion, CronusVLA WidowX moderate and aggressive reuse, and UniVLA WidowX task-aware fusion are the 16 September 2026 reruns on the RTX 5090. The SmolVLA reruns use the reconstructed SDPA evaluator, whose per-call time is about 5 percent lower than the original SmolVLA run on the same card, so their latency change against the original row includes that implementation difference.

## CogACT / WidowX

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 200 | 31 | 26 | +2.5 | 0.597 | yes |
| Foveation, keep 50% | 200 | 21 | 16 | +2.5 | 0.511 |  |
| Action repeat, k=2 | 200 | 6 | 82 | -38.0 | <0.001 | yes |
| Action repeat, k=4 | 200 | 3 | 94 | -45.5 | <0.001 |  |
| Depth pruning, 1 layer | 200 | 27 | 13 | +7.0 | 0.038 |  |
| Depth pruning, 2 layers | 200 | 34 | 15 | +9.5 | 0.009 | yes |
| Depth pruning, 4 layers | 200 | 37 | 32 | +2.5 | 0.630 |  |
| Guarded reuse, strict | 200 | 13 | 10 | +1.5 | 0.678 |  |
| Guarded reuse, moderate | 200 | 22 | 22 | +0.0 | 1.000 |  |
| Guarded reuse, aggressive | 200 | 18 | 12 | +3.0 | 0.362 | yes |
| Temporal fusion, motion-entropy | 200 | 29 | 13 | +8.0 | 0.019 | yes |
| Temporal fusion, task-aware | 200 | 29 | 13 | +8.0 | 0.019 |  |
| Temporal fusion, conservative-adaptive | 200 | 23 | 19 | +2.0 | 0.644 |  |

## CronusVLA / WidowX

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 200 | 16 | 11 | +2.5 | 0.442 | yes |
| Foveation, keep 50% | 200 | 10 | 10 | +0.0 | 1.000 |  |
| Action repeat, k=2 | 200 | 2 | 63 | -30.5 | <0.001 | yes |
| Action repeat, k=4 | 200 | 0 | 68 | -34.0 | <0.001 |  |
| Depth pruning, 1 layer | 200 | 13 | 10 | +1.5 | 0.678 |  |
| Depth pruning, 2 layers | 200 | 17 | 13 | +2.0 | 0.585 | yes |
| Depth pruning, 4 layers | 200 | 6 | 17 | -5.5 | 0.035 |  |
| Guarded reuse, strict | 200 | 9 | 5 | +2.0 | 0.424 |  |
| Guarded reuse, moderate | 200 | 10 | 9 | +0.5 | 1.000 |  |
| Guarded reuse, aggressive | 200 | 11 | 6 | +2.5 | 0.332 | yes |
| Temporal fusion, motion-entropy | 200 | 9 | 10 | -0.5 | 1.000 | yes |
| Temporal fusion, task-aware | 200 | 9 | 10 | -0.5 | 1.000 |  |
| Temporal fusion, conservative-adaptive | 200 | 9 | 10 | -0.5 | 1.000 |  |

## MiniVLA / WidowX

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 200 | 23 | 39 | -8.0 | 0.056 |  |
| Foveation, keep 50% | 200 | 28 | 30 | -1.0 | 0.896 | yes |
| Action repeat, k=2 | 200 | 15 | 27 | -6.0 | 0.088 | yes |
| Action repeat, k=4 | 200 | 26 | 40 | -7.0 | 0.109 |  |
| Depth pruning, 1 layer | 200 | 16 | 51 | -17.5 | <0.001 | yes |
| Depth pruning, 2 layers | 200 | 0 | 72 | -36.0 | <0.001 |  |
| Depth pruning, 4 layers | 200 | 0 | 72 | -36.0 | <0.001 |  |
| Guarded reuse, strict | 200 | 11 | 6 | +2.5 | 0.332 | yes |
| Guarded reuse, moderate | 200 | 13 | 13 | +0.0 | 1.000 |  |
| Guarded reuse, aggressive | 200 | 6 | 11 | -2.5 | 0.332 |  |
| Temporal fusion, motion-entropy | 200 | 20 | 14 | +3.0 | 0.392 | yes |
| Temporal fusion, task-aware | 200 | 19 | 20 | -0.5 | 1.000 |  |
| Temporal fusion, conservative-adaptive | 200 | 15 | 10 | +2.5 | 0.424 |  |

## OpenVLA / WidowX

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 200 | 15 | 57 | -21.0 | <0.001 |  |
| Foveation, keep 50% | 200 | 32 | 59 | -13.5 | 0.006 | yes |
| Action repeat, k=2 | 200 | 23 | 54 | -15.5 | <0.001 | yes |
| Action repeat, k=4 | 200 | 14 | 72 | -29.0 | <0.001 |  |
| Depth pruning, 1 layer | 200 | 32 | 32 | +0.0 | 1.000 |  |
| Depth pruning, 2 layers | 200 | 34 | 26 | +4.0 | 0.366 | yes |
| Depth pruning, 4 layers | 200 | 19 | 31 | -6.0 | 0.119 |  |
| Guarded reuse, strict | 200 | 12 | 8 | +2.0 | 0.503 | yes |
| Guarded reuse, moderate | 200 | 10 | 14 | -2.0 | 0.541 |  |
| Guarded reuse, aggressive | 200 | 18 | 23 | -2.5 | 0.533 |  |
| Temporal fusion, motion-entropy | 200 | 29 | 34 | -2.5 | 0.615 |  |
| Temporal fusion, task-aware | 200 | 41 | 27 | +7.0 | 0.114 | yes |
| Temporal fusion, conservative-adaptive | 200 | 18 | 9 | +4.5 | 0.122 |  |

## SpatialVLA / WidowX

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 200 | 21 | 11 | +5.0 | 0.110 | yes |
| Foveation, keep 50% | 200 | 19 | 20 | -0.5 | 1.000 |  |
| Action repeat, k=2 | 200 | 17 | 25 | -4.0 | 0.280 | yes |
| Action repeat, k=4 | 200 | 16 | 48 | -16.0 | <0.001 |  |
| Depth pruning, 1 layer | 200 | 8 | 21 | -6.5 | 0.024 | yes |
| Depth pruning, 2 layers | 200 | 4 | 38 | -17.0 | <0.001 |  |
| Depth pruning, 4 layers | 200 | 12 | 74 | -31.0 | <0.001 |  |
| Guarded reuse, strict | 200 | 0 | 0 | +0.0 | 1.000 |  |
| Guarded reuse, moderate | 200 | 1 | 1 | +0.0 | 1.000 |  |
| Guarded reuse, aggressive | 200 | 2 | 1 | +0.5 | 1.000 | yes |
| Temporal fusion, motion-entropy | 200 | 15 | 13 | +1.0 | 0.851 | yes |
| Temporal fusion, task-aware | 200 | 8 | 10 | -1.0 | 0.815 |  |
| Temporal fusion, conservative-adaptive | 200 | 1 | 1 | +0.0 | 1.000 |  |

## UniVLA / WidowX

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 200 | 11 | 28 | -8.5 | 0.009 | yes |
| Foveation, keep 50% | 200 | 9 | 38 | -14.5 | <0.001 |  |
| Action repeat, k=2 | 200 | 2 | 152 | -75.0 | <0.001 | yes |
| Action repeat, k=4 | 200 | 1 | 170 | -84.5 | <0.001 |  |
| Depth pruning, 1 layer | 200 | 8 | 19 | -5.5 | 0.052 |  |
| Depth pruning, 2 layers | 200 | 8 | 19 | -5.5 | 0.052 | yes |
| Depth pruning, 4 layers | 200 | 9 | 21 | -6.0 | 0.043 |  |
| Guarded reuse, strict | 200 | 0 | 0 | +0.0 | 1.000 |  |
| Guarded reuse, moderate | 200 | 0 | 0 | +0.0 | 1.000 | yes |
| Guarded reuse, aggressive | 200 | 0 | 2 | -1.0 | 0.500 |  |
| Temporal fusion, motion-entropy | 200 | 6 | 17 | -5.5 | 0.035 |  |
| Temporal fusion, task-aware | 200 | 9 | 17 | -4.0 | 0.169 |  |
| Temporal fusion, conservative-adaptive | 200 | 0 | 0 | +0.0 | 1.000 | yes |

## CogACT / Fractal

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 250 | 11 | 16 | -2.0 | 0.442 |  |
| Foveation, keep 50% | 250 | 12 | 11 | +0.4 | 1.000 | yes |
| Action repeat, k=2 | 250 | 16 | 19 | -1.2 | 0.736 | yes |
| Action repeat, k=4 | 250 | 23 | 70 | -18.8 | <0.001 |  |
| Depth pruning, 1 layer | 250 | 6 | 10 | -1.6 | 0.455 |  |
| Depth pruning, 2 layers | 250 | 7 | 10 | -1.2 | 0.629 |  |
| Depth pruning, 4 layers | 250 | 9 | 9 | +0.0 | 1.000 | yes |
| Guarded reuse, strict | 250 | 0 | 0 | +0.0 | 1.000 |  |
| Guarded reuse, moderate | 250 | 5 | 4 | +0.4 | 1.000 | yes |
| Guarded reuse, aggressive | 250 | 5 | 5 | +0.0 | 1.000 |  |
| Temporal fusion, motion-entropy | 250 | 4 | 9 | -2.0 | 0.267 |  |
| Temporal fusion, task-aware | 250 | 4 | 9 | -2.0 | 0.267 |  |
| Temporal fusion, conservative-adaptive | 250 | 1 | 4 | -1.2 | 0.375 | yes |

## CronusVLA / Fractal

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 250 | 9 | 12 | -1.2 | 0.664 |  |
| Foveation, keep 50% | 250 | 12 | 9 | +1.2 | 0.664 | yes |
| Action repeat, k=2 | 250 | 15 | 8 | +2.8 | 0.210 | yes |
| Action repeat, k=4 | 250 | 6 | 30 | -9.6 | <0.001 |  |
| Depth pruning, 1 layer | 250 | 9 | 7 | +0.8 | 0.804 | yes |
| Depth pruning, 2 layers | 250 | 4 | 14 | -4.0 | 0.031 |  |
| Depth pruning, 4 layers | 250 | 4 | 15 | -4.4 | 0.019 |  |
| Guarded reuse, strict | 250 | 6 | 9 | -1.2 | 0.607 |  |
| Guarded reuse, moderate | 250 | 4 | 6 | -0.8 | 0.754 |  |
| Guarded reuse, aggressive | 250 | 9 | 6 | +1.2 | 0.607 | yes |
| Temporal fusion, motion-entropy | 250 | 9 | 5 | +1.6 | 0.424 | yes |
| Temporal fusion, task-aware | 250 | 9 | 5 | +1.6 | 0.424 |  |
| Temporal fusion, conservative-adaptive | 250 | 9 | 5 | +1.6 | 0.424 |  |

## OpenVLA / Fractal

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 250 | 8 | 78 | -28.0 | <0.001 |  |
| Foveation, keep 50% | 250 | 22 | 50 | -11.2 | 0.001 | yes |
| Action repeat, k=2 | 250 | 38 | 27 | +4.4 | 0.214 | yes |
| Action repeat, k=4 | 250 | 34 | 34 | +0.0 | 1.000 |  |
| Depth pruning, 1 layer | 250 | 33 | 26 | +2.8 | 0.435 |  |
| Depth pruning, 2 layers | 250 | 37 | 24 | +5.2 | 0.124 | yes |
| Depth pruning, 4 layers | 250 | 38 | 35 | +1.2 | 0.815 |  |
| Guarded reuse, strict | 250 | 1 | 0 | +0.4 | 1.000 |  |
| Guarded reuse, moderate | 250 | 2 | 0 | +0.8 | 0.500 | yes |
| Guarded reuse, aggressive | 250 | 2 | 3 | -0.4 | 1.000 |  |
| Temporal fusion, motion-entropy | 250 | 39 | 23 | +6.4 | 0.056 |  |
| Temporal fusion, task-aware | 250 | 29 | 12 | +6.8 | 0.011 | yes |
| Temporal fusion, conservative-adaptive | 250 | 7 | 4 | +1.2 | 0.549 |  |

## SpatialVLA / Fractal

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 250 | 17 | 27 | -4.0 | 0.174 |  |
| Foveation, keep 50% | 250 | 16 | 21 | -2.0 | 0.511 | yes |
| Action repeat, k=2 | 250 | 33 | 33 | +0.0 | 1.000 | yes |
| Action repeat, k=4 | 250 | 19 | 93 | -29.6 | <0.001 |  |
| Depth pruning, 1 layer | 250 | 12 | 25 | -5.2 | 0.047 |  |
| Depth pruning, 2 layers | 250 | 17 | 23 | -2.4 | 0.430 | yes |
| Depth pruning, 4 layers | 250 | 7 | 47 | -16.0 | <0.001 |  |
| Guarded reuse, strict | 250 | 0 | 0 | +0.0 | 1.000 |  |
| Guarded reuse, moderate | 250 | 2 | 0 | +0.8 | 0.500 | yes |
| Guarded reuse, aggressive | 250 | 3 | 4 | -0.4 | 1.000 |  |
| Temporal fusion, motion-entropy | 250 | 17 | 20 | -1.2 | 0.743 |  |
| Temporal fusion, task-aware | 250 | 12 | 10 | +0.8 | 0.832 | yes |
| Temporal fusion, conservative-adaptive | 250 | 0 | 0 | +0.0 | 1.000 |  |

## OpenVLA / LIBERO Long

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 7 | 33 | -26.0 | <0.001 |  |
| Foveation, keep 50% | 100 | 8 | 20 | -12.0 | 0.036 | yes |
| Action repeat, k=2 | 100 | 10 | 20 | -10.0 | 0.099 | yes |
| Action repeat, k=4 | 100 | 11 | 35 | -24.0 | <0.001 |  |
| Depth pruning, 1 layer | 100 | 16 | 19 | -3.0 | 0.736 | yes |
| Depth pruning, 2 layers | 100 | 10 | 21 | -11.0 | 0.071 |  |
| Depth pruning, 4 layers | 100 | 9 | 39 | -30.0 | <0.001 |  |
| Guarded reuse, strict | 100 | 8 | 13 | -5.0 | 0.383 |  |
| Guarded reuse, moderate | 100 | 9 | 14 | -5.0 | 0.405 | yes |
| Guarded reuse, aggressive | 100 | 12 | 18 | -6.0 | 0.362 |  |
| Temporal fusion, motion-entropy | 100 | 9 | 16 | -7.0 | 0.230 | yes |
| Temporal fusion, task-aware | 100 | 10 | 18 | -8.0 | 0.185 |  |
| Temporal fusion, conservative-adaptive | 100 | 2 | 13 | -11.0 | 0.007 |  |

## SmolVLA / LIBERO Long

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 3 | 26 | -23.0 | <0.001 |  |
| Foveation, keep 50% | 100 | 6 | 27 | -21.0 | <0.001 | yes |
| Action repeat, k=2 | 100 | 9 | 20 | -11.0 | 0.061 | yes |
| Action repeat, k=4 | 100 | 5 | 27 | -22.0 | <0.001 |  |
| Depth pruning, 1 layer | 100 | 1 | 34 | -33.0 | <0.001 | yes |
| Depth pruning, 2 layers | 100 | 1 | 40 | -39.0 | <0.001 |  |
| Depth pruning, 4 layers | 100 | 0 | 42 | -42.0 | <0.001 |  |
| Guarded reuse, strict | 100 | 13 | 15 | -2.0 | 0.851 |  |
| Guarded reuse, moderate | 100 | 13 | 15 | -2.0 | 0.851 |  |
| Guarded reuse, aggressive | 100 | 13 | 13 | +0.0 | 1.000 | yes |
| Temporal fusion, motion-entropy | 100 | 12 | 11 | +1.0 | 1.000 |  |
| Temporal fusion, task-aware | 100 | 11 | 14 | -3.0 | 0.690 |  |
| Temporal fusion, conservative-adaptive | 100 | 14 | 11 | +3.0 | 0.690 | yes |

## UniVLA / LIBERO Long

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 8 | 8 | +0.0 | 1.000 |  |
| Foveation, keep 50% | 100 | 8 | 4 | +4.0 | 0.388 | yes |
| Action repeat, k=2 | 100 | 2 | 61 | -59.0 | <0.001 | yes |
| Action repeat, k=4 | 100 | 0 | 85 | -85.0 | <0.001 |  |
| Depth pruning, 1 layer | 100 | 8 | 6 | +2.0 | 0.790 |  |
| Depth pruning, 2 layers | 100 | 10 | 7 | +3.0 | 0.629 | yes |
| Depth pruning, 4 layers | 100 | 8 | 8 | +0.0 | 1.000 |  |
| Guarded reuse, strict | 100 | 0 | 0 | +0.0 | 1.000 | yes |
| Guarded reuse, moderate | 100 | 0 | 0 | +0.0 | 1.000 |  |
| Guarded reuse, aggressive | 100 | 0 | 0 | +0.0 | 1.000 |  |
| Temporal fusion, motion-entropy | 100 | 5 | 8 | -3.0 | 0.581 |  |
| Temporal fusion, task-aware | 100 | 7 | 6 | +1.0 | 1.000 | yes |
| Temporal fusion, conservative-adaptive | 100 | 0 | 0 | +0.0 | 1.000 |  |

## OpenVLA / LIBERO Goal

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 13 | 24 | -11.0 | 0.099 |  |
| Foveation, keep 50% | 100 | 10 | 10 | +0.0 | 1.000 | yes |
| Action repeat, k=2 | 100 | 10 | 17 | -7.0 | 0.248 | yes |
| Action repeat, k=4 | 100 | 6 | 21 | -15.0 | 0.006 |  |
| Depth pruning, 1 layer | 100 | 12 | 15 | -3.0 | 0.701 | yes |
| Depth pruning, 2 layers | 100 | 6 | 22 | -16.0 | 0.004 |  |
| Depth pruning, 4 layers | 100 | 4 | 38 | -34.0 | <0.001 |  |
| Guarded reuse, strict | 100 | 2 | 6 | -4.0 | 0.289 |  |
| Guarded reuse, moderate | 100 | 4 | 10 | -6.0 | 0.180 |  |
| Guarded reuse, aggressive | 100 | 8 | 11 | -3.0 | 0.648 | yes |
| Temporal fusion, motion-entropy | 100 | 7 | 8 | -1.0 | 1.000 |  |
| Temporal fusion, task-aware | 100 | 10 | 12 | -2.0 | 0.832 |  |
| Temporal fusion, conservative-adaptive | 100 | 5 | 3 | +2.0 | 0.727 | yes |

## SmolVLA / LIBERO Goal

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 7 | 29 | -22.0 | <0.001 |  |
| Foveation, keep 50% | 100 | 6 | 19 | -13.0 | 0.015 | yes |
| Action repeat, k=2 | 100 | 7 | 15 | -8.0 | 0.134 | yes |
| Action repeat, k=4 | 100 | 6 | 27 | -21.0 | <0.001 |  |
| Depth pruning, 1 layer | 100 | 6 | 24 | -18.0 | 0.001 | yes |
| Depth pruning, 2 layers | 100 | 5 | 32 | -27.0 | <0.001 |  |
| Depth pruning, 4 layers | 100 | 2 | 54 | -52.0 | <0.001 |  |
| Guarded reuse, strict | 100 | 12 | 15 | -3.0 | 0.701 |  |
| Guarded reuse, moderate | 100 | 11 | 15 | -4.0 | 0.557 |  |
| Guarded reuse, aggressive | 100 | 12 | 13 | -1.0 | 1.000 | yes |
| Temporal fusion, motion-entropy | 100 | 12 | 15 | -3.0 | 0.701 |  |
| Temporal fusion, task-aware | 100 | 11 | 17 | -6.0 | 0.345 |  |
| Temporal fusion, conservative-adaptive | 100 | 14 | 16 | -2.0 | 0.856 | yes |

## UniVLA / LIBERO Goal

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 3 | 8 | -5.0 | 0.227 |  |
| Foveation, keep 50% | 100 | 8 | 1 | +7.0 | 0.039 | yes |
| Action repeat, k=2 | 100 | 3 | 61 | -58.0 | <0.001 | yes |
| Action repeat, k=4 | 100 | 0 | 92 | -92.0 | <0.001 |  |
| Depth pruning, 1 layer | 100 | 6 | 5 | +1.0 | 1.000 | yes |
| Depth pruning, 2 layers | 100 | 7 | 10 | -3.0 | 0.629 |  |
| Depth pruning, 4 layers | 100 | 6 | 14 | -8.0 | 0.115 |  |
| Guarded reuse, strict | 100 | 1 | 0 | +1.0 | 1.000 | yes |
| Guarded reuse, moderate | 100 | 1 | 0 | +1.0 | 1.000 |  |
| Guarded reuse, aggressive | 100 | 0 | 0 | +0.0 | 1.000 |  |
| Temporal fusion, motion-entropy | 100 | 7 | 4 | +3.0 | 0.549 | yes |
| Temporal fusion, task-aware | 100 | 6 | 5 | +1.0 | 1.000 |  |
| Temporal fusion, conservative-adaptive | 100 | 0 | 0 | +0.0 | 1.000 |  |

## OpenVLA / LIBERO Object

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 8 | 21 | -13.0 | 0.024 |  |
| Foveation, keep 50% | 100 | 10 | 11 | -1.0 | 1.000 | yes |
| Action repeat, k=2 | 100 | 10 | 13 | -3.0 | 0.678 | yes |
| Action repeat, k=4 | 100 | 6 | 34 | -28.0 | <0.001 |  |
| Depth pruning, 1 layer | 100 | 10 | 12 | -2.0 | 0.832 | yes |
| Depth pruning, 2 layers | 100 | 8 | 16 | -8.0 | 0.152 |  |
| Depth pruning, 4 layers | 100 | 7 | 29 | -22.0 | <0.001 |  |
| Guarded reuse, strict | 100 | 4 | 6 | -2.0 | 0.754 | yes |
| Guarded reuse, moderate | 100 | 4 | 11 | -7.0 | 0.118 |  |
| Guarded reuse, aggressive | 100 | 6 | 8 | -2.0 | 0.790 |  |
| Temporal fusion, motion-entropy | 100 | 7 | 11 | -4.0 | 0.481 |  |
| Temporal fusion, task-aware | 100 | 9 | 10 | -1.0 | 1.000 | yes |
| Temporal fusion, conservative-adaptive | 100 | 2 | 7 | -5.0 | 0.180 |  |

## SmolVLA / LIBERO Object

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 5 | 25 | -20.0 | <0.001 |  |
| Foveation, keep 50% | 100 | 2 | 15 | -13.0 | 0.002 | yes |
| Action repeat, k=2 | 100 | 5 | 9 | -4.0 | 0.424 | yes |
| Action repeat, k=4 | 100 | 1 | 24 | -23.0 | <0.001 |  |
| Depth pruning, 1 layer | 100 | 4 | 25 | -21.0 | <0.001 | yes |
| Depth pruning, 2 layers | 100 | 2 | 56 | -54.0 | <0.001 |  |
| Depth pruning, 4 layers | 100 | 0 | 81 | -81.0 | <0.001 |  |
| Guarded reuse, strict | 100 | 4 | 9 | -5.0 | 0.267 |  |
| Guarded reuse, moderate | 100 | 4 | 9 | -5.0 | 0.267 |  |
| Guarded reuse, aggressive | 100 | 5 | 9 | -4.0 | 0.424 | yes |
| Temporal fusion, motion-entropy | 100 | 5 | 9 | -4.0 | 0.424 |  |
| Temporal fusion, task-aware | 100 | 5 | 9 | -4.0 | 0.424 | yes |
| Temporal fusion, conservative-adaptive | 100 | 5 | 13 | -8.0 | 0.096 |  |

## UniVLA / LIBERO Object

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 4 | 0 | +4.0 | 0.125 | yes |
| Foveation, keep 50% | 100 | 3 | 2 | +1.0 | 1.000 |  |
| Action repeat, k=2 | 100 | 1 | 76 | -75.0 | <0.001 | yes |
| Action repeat, k=4 | 100 | 0 | 95 | -95.0 | <0.001 |  |
| Depth pruning, 1 layer | 100 | 4 | 1 | +3.0 | 0.375 | yes |
| Depth pruning, 2 layers | 100 | 4 | 6 | -2.0 | 0.754 |  |
| Depth pruning, 4 layers | 100 | 4 | 8 | -4.0 | 0.388 |  |
| Guarded reuse, strict | 100 | 0 | 0 | +0.0 | 1.000 | yes |
| Guarded reuse, moderate | 100 | 0 | 0 | +0.0 | 1.000 |  |
| Guarded reuse, aggressive | 100 | 0 | 0 | +0.0 | 1.000 |  |
| Temporal fusion, motion-entropy | 100 | 3 | 1 | +2.0 | 0.625 | yes |
| Temporal fusion, task-aware | 100 | 3 | 2 | +1.0 | 1.000 |  |
| Temporal fusion, conservative-adaptive | 100 | 0 | 0 | +0.0 | 1.000 |  |

## OpenVLA / LIBERO Spatial

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 9 | 32 | -23.0 | <0.001 |  |
| Foveation, keep 50% | 100 | 14 | 9 | +5.0 | 0.405 | yes |
| Action repeat, k=2 | 100 | 8 | 14 | -6.0 | 0.286 | yes |
| Action repeat, k=4 | 100 | 7 | 38 | -31.0 | <0.001 |  |
| Depth pruning, 1 layer | 100 | 10 | 26 | -16.0 | 0.011 | yes |
| Depth pruning, 2 layers | 100 | 9 | 31 | -22.0 | <0.001 |  |
| Depth pruning, 4 layers | 100 | 7 | 37 | -30.0 | <0.001 |  |
| Guarded reuse, strict | 100 | 5 | 1 | +4.0 | 0.219 | yes |
| Guarded reuse, moderate | 100 | 6 | 8 | -2.0 | 0.790 |  |
| Guarded reuse, aggressive | 100 | 10 | 11 | -1.0 | 1.000 |  |
| Temporal fusion, motion-entropy | 100 | 12 | 5 | +7.0 | 0.143 | yes |
| Temporal fusion, task-aware | 100 | 12 | 13 | -1.0 | 1.000 |  |
| Temporal fusion, conservative-adaptive | 100 | 5 | 4 | +1.0 | 1.000 |  |

## SmolVLA / LIBERO Spatial

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 13 | 26 | -13.0 | 0.053 |  |
| Foveation, keep 50% | 100 | 12 | 12 | +0.0 | 1.000 | yes |
| Action repeat, k=2 | 100 | 13 | 25 | -12.0 | 0.073 | yes |
| Action repeat, k=4 | 100 | 7 | 48 | -41.0 | <0.001 |  |
| Depth pruning, 1 layer | 100 | 12 | 26 | -14.0 | 0.034 | yes |
| Depth pruning, 2 layers | 100 | 10 | 48 | -38.0 | <0.001 |  |
| Depth pruning, 4 layers | 100 | 6 | 53 | -47.0 | <0.001 |  |
| Guarded reuse, strict | 100 | 12 | 10 | +2.0 | 0.832 | yes |
| Guarded reuse, moderate | 100 | 13 | 11 | +2.0 | 0.839 |  |
| Guarded reuse, aggressive | 100 | 13 | 13 | +0.0 | 1.000 |  |
| Temporal fusion, motion-entropy | 100 | 12 | 7 | +5.0 | 0.359 |  |
| Temporal fusion, task-aware | 100 | 14 | 8 | +6.0 | 0.286 | yes |
| Temporal fusion, conservative-adaptive | 100 | 13 | 11 | +2.0 | 0.839 |  |

## UniVLA / LIBERO Spatial

| Setting | n | fail->success | success->fail | delta (pts) | p | table |
|---|---|---|---|---|---|---|
| Foveation, keep 20% | 100 | 1 | 5 | -4.0 | 0.219 |  |
| Foveation, keep 50% | 100 | 5 | 4 | +1.0 | 1.000 | yes |
| Action repeat, k=2 | 100 | 0 | 68 | -68.0 | <0.001 | yes |
| Action repeat, k=4 | 100 | 0 | 95 | -95.0 | <0.001 |  |
| Depth pruning, 1 layer | 100 | 5 | 4 | +1.0 | 1.000 | yes |
| Depth pruning, 2 layers | 100 | 5 | 6 | -1.0 | 1.000 |  |
| Depth pruning, 4 layers | 100 | 4 | 5 | -1.0 | 1.000 |  |
| Guarded reuse, strict | 100 | 0 | 0 | +0.0 | 1.000 | yes |
| Guarded reuse, moderate | 100 | 0 | 0 | +0.0 | 1.000 |  |
| Guarded reuse, aggressive | 100 | 0 | 0 | +0.0 | 1.000 |  |
| Temporal fusion, motion-entropy | 100 | 1 | 5 | -4.0 | 0.219 |  |
| Temporal fusion, task-aware | 100 | 1 | 4 | -3.0 | 0.375 |  |
| Temporal fusion, conservative-adaptive | 100 | 0 | 0 | +0.0 | 1.000 | yes |
