# Model Expansion — Run Tables

Rows are conditions, columns are cells (model × benchmark). Values are the
change in success rate against that column's own baseline, on paired episodes.
**Bold** = passes the current correction (α ≈ 0.0013). `Not` = to be run.
`—` = no public checkpoint.

---

## 1. SimplerEnv — WidowX-Bridge (96 episodes) and Google Robot / Fractal (135 episodes)

| | OpenVLA<br>Bridge | OpenVLA<br>Fractal | SpatialVLA<br>Bridge | SpatialVLA<br>Fractal | UniVLA<br>Bridge | UniVLA<br>Fractal | TurboVLA<br>Bridge | TurboVLA<br>Fractal | CoTinyVLA<br>Bridge | CoTinyVLA<br>Fractal | FLOWER<br>Bridge | FLOWER<br>Fractal | MiniVLA<br>Bridge | MiniVLA<br>Fractal | SmolVLA<br>Bridge | SmolVLA<br>Fractal |
|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| *params* | *7B* | *7B* | *4B* | *4B* | *8.5B* | *8.5B* | *0.2B* | *0.2B* | *0.9B* | *0.9B* | *1B* | *1B* | *1B* | *1B* | *4B* | *4B* |
| *decoder layers* | *32* | *32* | *26* | *26* | *32* | *32* | *?* | *?* | *?* | *?* | *?* | *?* | *?* | *?* | *?* | *?* |
| *baseline success* | *15.6%* | *38.5%* | *30.2%* | *84.4%* | *81.2%* | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| action repeat 2 | −8.3 | +5.2 | +12.5 | ±0.0 | **−69.8** | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| action repeat 4 | **−11.5** | −1.5 | −12.5 | **−40.0** | **−81.2** | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| foveation log-polar | +18.8 | **−19.3** | −8.3 | +0.7 | +5.2 | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| foveation blur | +17.7 | −8.9 | ±0.0 | −1.5 | −8.3 | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 1 | +2.1 | +0.7 | −10.4 | +8.1 | −3.1 | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 2 | ±0.0 | ±0.0 | −9.4 | +3.0 | −4.2 | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 4 | +1.0 | **+15.6** | **−28.1** | **−17.8** | −2.1 | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |

10 cells × 8 conditions = **80 runs, 9,240 episodes.**

---

## 2. LIBERO — Spatial, Object, Goal, Long (= LIBERO-10), 50 episodes each

| | TurboVLA<br>Spatial | TurboVLA<br>Object | TurboVLA<br>Goal | TurboVLA<br>Long | CoTinyVLA<br>Spatial | CoTinyVLA<br>Object | CoTinyVLA<br>Goal | CoTinyVLA<br>Long | FLOWER<br>Spatial | FLOWER<br>Object | FLOWER<br>Goal | FLOWER<br>Long | MiniVLA<br>Spatial | MiniVLA<br>Object | MiniVLA<br>Goal | MiniVLA<br>Long | SmolVLA<br>Spatial | SmolVLA<br>Object | SmolVLA<br>Goal | SmolVLA<br>Long |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| *params* | *0.2B* | *0.2B* | *0.2B* | *0.2B* | *0.9B* | *0.9B* | *0.9B* | *0.9B* | *1B* | *1B* | *1B* | *1B* | *1B* | *1B* | *1B* | *1B* | *4B* | *4B* | *4B* | *4B* |
| *LIBERO checkpoint?* | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| *baseline success* | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| action repeat 2 | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| action repeat 4 | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| foveation log-polar | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| foveation blur | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 1 | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 2 | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 4 | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |

20 cells × 8 conditions = **160 runs, 8,000 episodes.**

LIBERO cannot be run zero-shot, and each released checkpoint covers one suite
only, so the `LIBERO checkpoint?` row decides how many columns a model can
fill. Long and LIBERO-10 are the same suite; the fourth column here is Goal.

---

**Total: 30 cells, 240 runs, 17,240 episodes.**
