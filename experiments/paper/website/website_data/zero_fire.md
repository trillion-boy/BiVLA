# Zero-fire episodes of the guarded reuse settings (run-to-run noise)

A zero-fire episode is one on which the reuse gate never fired (`reuses` = 0 in episodes.jsonl), so the original policy was called at every step. Its outcome and length are compared with the original run of the same (task, episode index). `differs` is the share of zero-fire episodes whose success differs from the original; `steps identical` is the share whose step count equals the original's; `both-success steps identical` restricts the latter to episodes that succeed in both runs (the definition used in the setup.tex comment). Percentages to one decimal.

## Pooled over the three presets, per pair

| backbone | env | guarded-reuse episodes | zero-fire | zero-fire % | differs % | steps identical % | both-success steps identical % | setup.tex comment (differs %) |
|---|---|---|---|---|---|---|---|---|
| CogACT | WidowX | 600 | 522 | 87.0 | 16.5 | 53.1 | 36.9 | 16.5 |
| OpenVLA | WidowX | 600 | 409 | 68.2 | 13.4 | 61.4 | 51.9 | 13.4 |
| SpatialVLA | WidowX | 600 | 572 | 95.3 | 0.5 | 99.3 | 99.6 | 0 or 0.5 |
| CronusVLA | WidowX | 600 | 538 | 89.7 | 9.3 | 66.9 | 26.9 | 11.3 |
| UniVLA | WidowX | 600 | 598 | 99.7 | 0.3 | 96.8 | 96.7 | 0.2 to 0.3 |
| MiniVLA | WidowX | 600 | 438 | 73.0 | 9.1 | 80.1 | 74.6 | 9.1 |
| CogACT | Fractal | 750 | 716 | 95.5 | 1.8 | 84.2 | 78.7 | 1.8 |
| OpenVLA | Fractal | 750 | 612 | 81.6 | 0.0 | 99.7 | 99.2 | 0 |
| SpatialVLA | Fractal | 750 | 604 | 80.5 | 0.0 | 100.0 | 100.0 | 0 or 0.5 |
| CronusVLA | Fractal | 750 | 609 | 81.2 | 4.6 | 46.0 | 18.4 | 4.6 |
| OpenVLA | LIBERO Long | 300 | 16 | 5.3 | 0.0 | 100.0 | 100.0 | 0 (LIBERO) |
| UniVLA | LIBERO Long | 300 | 286 | 95.3 | 0.0 | 95.8 | 95.2 | 0.2 to 0.3 |
| SmolVLA | LIBERO Long | 300 | 118 | 39.3 | 30.5 | 33.9 | 4.5 | 19.9 (SmolVLA pooled) |
| OpenVLA | LIBERO Goal | 300 | 52 | 17.3 | 0.0 | 100.0 | 100.0 | 0 (LIBERO) |
| UniVLA | LIBERO Goal | 300 | 300 | 100.0 | 0.7 | 98.0 | 98.6 | 0.2 to 0.3 |
| SmolVLA | LIBERO Goal | 300 | 203 | 67.7 | 23.2 | 7.9 | 2.8 | 19.9 (SmolVLA pooled) |
| OpenVLA | LIBERO Object | 300 | 17 | 5.7 | 0.0 | 100.0 | 100.0 | 0 (LIBERO) |
| UniVLA | LIBERO Object | 300 | 300 | 100.0 | 0.0 | 100.0 | 100.0 | 0.2 to 0.3 |
| SmolVLA | LIBERO Object | 300 | 219 | 73.0 | 10.5 | 5.0 | 5.1 | 19.9 (SmolVLA pooled) |
| OpenVLA | LIBERO Spatial | 300 | 59 | 19.7 | 0.0 | 100.0 | 100.0 | 0 (LIBERO) |
| UniVLA | LIBERO Spatial | 300 | 300 | 100.0 | 0.0 | 100.0 | 100.0 | 0.2 to 0.3 |
| SmolVLA | LIBERO Spatial | 300 | 220 | 73.3 | 21.8 | 10.0 | 2.0 | 19.9 (SmolVLA pooled) |

## Pooled over environments where the setup.tex comment pools

| pool | zero-fire | differs % | both-success steps identical % |
|---|---|---|---|
| OpenVLA LIBERO (4 suites) | 144 | 0.0 | 100.0 |
| SmolVLA LIBERO (4 suites) | 760 | 20.3 | 3.5 |
| UniVLA LIBERO (4 suites) | 1186 | 0.2 | 98.5 |
| UniVLA all (5 pairs) | 1784 | 0.2 | 98.0 |
| SpatialVLA (2 pairs) | 1176 | 0.3 | 99.8 |

## Per preset

| backbone | env | preset | episodes | zero-fire | differs | differs % | steps identical % | both-success steps identical % | avg reuses | cell latency change vs original % |
|---|---|---|---|---|---|---|---|---|---|---|
| CogACT | WidowX | strict | 200 | 189 | 20 | 10.6 | 69.3 | 57.8 | 0.16 | -0.20 |
| CogACT | WidowX | moderate | 200 | 174 | 40 | 23.0 | 45.4 | 28.6 | 0.48 | -0.77 |
| CogACT | WidowX | aggressive | 200 | 159 | 26 | 16.4 | 42.1 | 22.4 | 1.35 | -1.95 |
| OpenVLA | WidowX | strict | 200 | 149 | 16 | 10.7 | 69.8 | 63.3 | 1.84 | -3.58 |
| OpenVLA | WidowX | moderate | 200 | 132 | 15 | 11.4 | 62.9 | 53.4 | 2.51 | -4.19 |
| OpenVLA | WidowX | aggressive | 200 | 128 | 24 | 18.8 | 50.0 | 35.5 | 3.31 | -5.06 |
| SpatialVLA | WidowX | strict | 200 | 198 | 0 | 0.0 | 100.0 | 100.0 | 0.01 | -0.22 |
| SpatialVLA | WidowX | moderate | 200 | 194 | 2 | 1.0 | 98.5 | 98.9 | 0.03 | +0.36 |
| SpatialVLA | WidowX | aggressive | 200 | 180 | 1 | 0.6 | 99.4 | 100.0 | 0.19 | -0.22 |
| CronusVLA | WidowX | strict | 200 | 195 | 14 | 7.2 | 70.3 | 28.6 | 0.03 | -0.30 |
| CronusVLA | WidowX | moderate | 200 | 181 | 19 | 10.5 | 66.9 | 27.1 | 0.17 | -2.40 |
| CronusVLA | WidowX | aggressive | 200 | 162 | 17 | 10.5 | 63.0 | 25.0 | 0.69 | -2.95 |
| UniVLA | WidowX | strict | 200 | 200 | 0 | 0.0 | 100.0 | 100.0 | 0.00 | +0.11 |
| UniVLA | WidowX | moderate | 200 | 199 | 0 | 0.0 | 95.5 | 94.9 | 0.01 | -0.69 |
| UniVLA | WidowX | aggressive | 200 | 199 | 2 | 1.0 | 95.0 | 95.4 | 0.01 | -0.14 |
| MiniVLA | WidowX | strict | 200 | 159 | 13 | 8.2 | 82.4 | 77.3 | 1.96 | -2.63 |
| MiniVLA | WidowX | moderate | 200 | 144 | 17 | 11.8 | 77.1 | 72.9 | 2.71 | -2.11 |
| MiniVLA | WidowX | aggressive | 200 | 135 | 10 | 7.4 | 80.7 | 73.3 | 2.64 | -3.09 |
| CogACT | Fractal | strict | 250 | 249 | 0 | 0.0 | 97.2 | 95.8 | 0.00 | -0.18 |
| CogACT | Fractal | moderate | 250 | 239 | 6 | 2.5 | 77.8 | 69.9 | 0.05 | +0.56 |
| CogACT | Fractal | aggressive | 250 | 228 | 7 | 3.1 | 76.8 | 69.1 | 0.25 | +4.32 |
| OpenVLA | Fractal | strict | 250 | 213 | 0 | 0.0 | 100.0 | 100.0 | 1.46 | -1.66 |
| OpenVLA | Fractal | moderate | 250 | 205 | 0 | 0.0 | 99.5 | 98.8 | 1.60 | -1.28 |
| OpenVLA | Fractal | aggressive | 250 | 194 | 0 | 0.0 | 99.5 | 98.8 | 2.49 | -2.09 |
| SpatialVLA | Fractal | strict | 250 | 233 | 0 | 0.0 | 100.0 | 100.0 | 0.26 | -0.74 |
| SpatialVLA | Fractal | moderate | 250 | 206 | 0 | 0.0 | 100.0 | 100.0 | 0.41 | -0.68 |
| SpatialVLA | Fractal | aggressive | 250 | 165 | 0 | 0.0 | 100.0 | 100.0 | 1.20 | -1.99 |
| CronusVLA | Fractal | strict | 250 | 230 | 15 | 6.5 | 48.3 | 18.1 | 0.36 | -1.29 |
| CronusVLA | Fractal | moderate | 250 | 209 | 5 | 2.4 | 47.4 | 17.3 | 0.88 | -1.39 |
| CronusVLA | Fractal | aggressive | 250 | 170 | 8 | 4.7 | 41.2 | 20.0 | 2.14 | -3.08 |
| OpenVLA | LIBERO Long | strict | 100 | 8 | 0 | 0.0 | 100.0 | 100.0 | 14.37 | -2.75 |
| OpenVLA | LIBERO Long | moderate | 100 | 5 | 0 | 0.0 | 100.0 | 100.0 | 22.13 | -4.54 |
| OpenVLA | LIBERO Long | aggressive | 100 | 3 | 0 | 0.0 | 100.0 | n/a | 40.47 | -8.90 |
| UniVLA | LIBERO Long | strict | 100 | 97 | 0 | 0.0 | 96.9 | 96.4 | 0.05 | -0.64 |
| UniVLA | LIBERO Long | moderate | 100 | 97 | 0 | 0.0 | 96.9 | 96.4 | 0.06 | -0.01 |
| UniVLA | LIBERO Long | aggressive | 100 | 92 | 0 | 0.0 | 93.5 | 92.7 | 0.17 | -2.73 |
| SmolVLA | LIBERO Long | strict | 100 | 74 | 21 | 28.4 | 37.8 | 3.8 | 0.58 | -5.54 |
| SmolVLA | LIBERO Long | moderate | 100 | 36 | 13 | 36.1 | 30.6 | 7.7 | 2.33 | -5.16 |
| SmolVLA | LIBERO Long | aggressive | 100 | 8 | 2 | 25.0 | 12.5 | 0.0 | 11.17 | -7.10 |
| OpenVLA | LIBERO Goal | strict | 100 | 31 | 0 | 0.0 | 100.0 | 100.0 | 7.91 | -4.61 |
| OpenVLA | LIBERO Goal | moderate | 100 | 13 | 0 | 0.0 | 100.0 | 100.0 | 8.48 | -4.88 |
| OpenVLA | LIBERO Goal | aggressive | 100 | 8 | 0 | 0.0 | 100.0 | 100.0 | 18.29 | -10.37 |
| UniVLA | LIBERO Goal | strict | 100 | 100 | 1 | 1.0 | 98.0 | 98.9 | 0.00 | -0.72 |
| UniVLA | LIBERO Goal | moderate | 100 | 100 | 1 | 1.0 | 97.0 | 97.8 | 0.00 | -1.57 |
| UniVLA | LIBERO Goal | aggressive | 100 | 100 | 0 | 0.0 | 99.0 | 98.9 | 0.00 | -1.44 |
| SmolVLA | LIBERO Goal | strict | 100 | 90 | 22 | 24.4 | 8.9 | 3.2 | 0.19 | -6.32 |
| SmolVLA | LIBERO Goal | moderate | 100 | 74 | 17 | 23.0 | 9.5 | 3.8 | 0.71 | -6.79 |
| SmolVLA | LIBERO Goal | aggressive | 100 | 39 | 8 | 20.5 | 2.6 | 0.0 | 3.22 | -8.54 |
| OpenVLA | LIBERO Object | strict | 100 | 14 | 0 | 0.0 | 100.0 | 100.0 | 6.09 | -2.62 |
| OpenVLA | LIBERO Object | moderate | 100 | 3 | 0 | 0.0 | 100.0 | 100.0 | 9.51 | -5.64 |
| OpenVLA | LIBERO Object | aggressive | 100 | 0 | 0 | n/a | n/a | n/a | 16.08 | -8.79 |
| UniVLA | LIBERO Object | strict | 100 | 100 | 0 | 0.0 | 100.0 | 100.0 | 0.00 | +0.48 |
| UniVLA | LIBERO Object | moderate | 100 | 100 | 0 | 0.0 | 100.0 | 100.0 | 0.00 | +0.64 |
| UniVLA | LIBERO Object | aggressive | 100 | 100 | 0 | 0.0 | 100.0 | 100.0 | 0.00 | -0.24 |
| SmolVLA | LIBERO Object | strict | 100 | 91 | 9 | 9.9 | 5.5 | 4.9 | 0.12 | -5.32 |
| SmolVLA | LIBERO Object | moderate | 100 | 82 | 8 | 9.8 | 4.9 | 5.4 | 0.37 | -5.52 |
| SmolVLA | LIBERO Object | aggressive | 100 | 46 | 6 | 13.0 | 4.3 | 5.0 | 2.33 | -6.24 |
| OpenVLA | LIBERO Spatial | strict | 100 | 37 | 0 | 0.0 | 100.0 | 100.0 | 2.62 | -1.93 |
| OpenVLA | LIBERO Spatial | moderate | 100 | 15 | 0 | 0.0 | 100.0 | 100.0 | 4.79 | -3.30 |
| OpenVLA | LIBERO Spatial | aggressive | 100 | 7 | 0 | 0.0 | 100.0 | 100.0 | 12.40 | -7.76 |
| UniVLA | LIBERO Spatial | strict | 100 | 100 | 0 | 0.0 | 100.0 | 100.0 | 0.00 | +1.41 |
| UniVLA | LIBERO Spatial | moderate | 100 | 100 | 0 | 0.0 | 100.0 | 100.0 | 0.00 | +1.04 |
| UniVLA | LIBERO Spatial | aggressive | 100 | 100 | 0 | 0.0 | 100.0 | 100.0 | 0.00 | +0.44 |
| SmolVLA | LIBERO Spatial | strict | 100 | 97 | 22 | 22.7 | 12.4 | 1.6 | 0.03 | -3.47 |
| SmolVLA | LIBERO Spatial | moderate | 100 | 80 | 18 | 22.5 | 10.0 | 1.8 | 0.33 | -4.35 |
| SmolVLA | LIBERO Spatial | aggressive | 100 | 43 | 8 | 18.6 | 4.7 | 2.9 | 1.82 | -5.12 |

## Cells whose gate never fired (10)

Paper (Sec. IV-A): ten cells, all on UniVLA, differ from their original row by up to 1.6 percent in latency.

| backbone | env | preset | latency ms/step | original | change % | in table |
|---|---|---|---|---|---|---|
| UniVLA | WidowX | strict | 191.07 | 190.86 | +0.11 |  |
| UniVLA | LIBERO Goal | strict | 64.67 | 65.14 | -0.72 | yes |
| UniVLA | LIBERO Goal | moderate | 64.12 | 65.14 | -1.57 |  |
| UniVLA | LIBERO Goal | aggressive | 64.20 | 65.14 | -1.44 |  |
| UniVLA | LIBERO Object | strict | 62.71 | 62.41 | +0.48 | yes |
| UniVLA | LIBERO Object | moderate | 62.81 | 62.41 | +0.64 |  |
| UniVLA | LIBERO Object | aggressive | 62.26 | 62.41 | -0.24 |  |
| UniVLA | LIBERO Spatial | strict | 73.27 | 72.25 | +1.41 | yes |
| UniVLA | LIBERO Spatial | moderate | 73.00 | 72.25 | +1.04 |  |
| UniVLA | LIBERO Spatial | aggressive | 72.57 | 72.25 | +0.44 |  |

## Comparison with the paper text (Sec. IV-A) and the setup.tex comment of 2026-09-15

| statement | paper | reproduced (pooled over the three presets, per pair) |
|---|---|---|
| outcome differs, CogACT, CronusVLA, MiniVLA and OpenVLA WidowX (6 pairs) | 2 to 17 percent | 1.8 to 16.5 percent |
| outcome differs, SpatialVLA, UniVLA, OpenVLA Fractal and LIBERO (12 pairs) | at most one episode in a hundred | 0.0 to 0.7 percent |
| outcome differs, SmolVLA reuse cells (4 pairs) | 10 to 30 percent (Hardware paragraph) | 10.5 to 30.5 percent (per preset 9.8 to 36.1) |
| both-success zero-fire episodes with identical steps, reproducing pairs | 97 to 100 percent (comment) | 95.2 to 100.0 percent |
| both-success zero-fire episodes with identical steps, the other six pairs | 18 to 79 percent (comment) | 18.4 to 78.7 percent |
| SmolVLA pooled outcome difference | 19.9 percent (comment) | 20.3 percent |
| cells whose gate never fired | ten, all UniVLA, latency within 1.6 percent of the original | 10, all UniVLA, -1.57 to +1.41 percent |

Remarks (data only):

- The per-pair comment values match the reproduced values except CronusVLA WidowX (comment 11.3, reproduced 9.3): the comment is dated 2026-09-15 and the moderate and aggressive presets of that pair were rerun on 2026-09-16 (records carry the RTX 5090 tag), so the comment describes the superseded runs.
- The comment's 0.5 for SpatialVLA is the WidowX pair; Fractal is 0.0. UniVLA LIBERO Goal is 0.7 (2 of 300), above the comment's 0.2 to 0.3 range, which matches the UniVLA pools (0.2).
- UniVLA WidowX and UniVLA Long both-success identical-step shares are 96.7 and 95.2, below the comment's 97; the UniVLA pool over all five pairs is 98.0.
- OpenVLA LIBERO cells have few zero-fire episodes (0 to 37 per cell) because the gate fires on most episodes there; OpenVLA Object aggressive has none.
