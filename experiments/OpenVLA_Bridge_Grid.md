# OpenVLA on SimplerEnv WidowX-Bridge — the full condition grid

Runs of 2026-08-05, `openvla/openvla-7b`, `unnorm_key=bridge_orig`,
`attn_implementation=eager`, 4 tasks × N=24 = **96 episodes per condition**,
Colab. Every condition replays env ids 0–23 per task, so all comparisons below
are **paired** (McNemar exact test on the discordant pairs,
`adaptive_sparse_vla/paired_test.py`).

## Results

| condition | eggplant | carrot | stack | spoon | **avg** | Δ | 95% CI | **p** | ms/infer | ms/env-step |
|---|---|---|---|---|---|---|---|---|---|---|
| **Original policy** | 25.0 | 16.7 | 12.5 | 8.3 | **15.6%** | — | — | — | 518 | 518 |
| **fixed foveation** (log-polar 20%) ¹ | 33.3 | 16.7 | 45.8 | 41.7 | **34.4%** | **+18.8** | [+6.7, +30.8] | **0.0051** | 518 | 518 |
| **fixed foveation** (blur 20%) | 62.5 | 25.0 | 20.8 | 25.0 | **33.3%** | **+17.7** | [+6.2, +29.3] | **0.0060** | 518 | 518 |
| action repeat 2 | 12.5 | 4.2 | 8.3 | 4.2 | 7.3% | −8.3 | [−15.8, −0.9] | 0.0574 | 519 | 260 (2.0×) |
| **action repeat 4** | 12.5 | 0.0 | 0.0 | 4.2 | **4.2%** | **−11.5** | [−17.8, −5.1] | **0.0010** | 522 | **131 (4.0×)** |
| fixed depth pruning (1 of 32) | 41.7 | 12.5 | 12.5 | 4.2 | 17.7% | +2.1 | [−6.1, +10.2] | 0.80 | 504 (1.03×) | 504 |
| fixed depth pruning (4 of 32) | 45.8 | 8.3 | 12.5 | 0.0 | 16.7% | +1.0 | [−8.3, +10.4] | 1.00 | 460 (1.13×) | 460 |
| fixed depth pruning (8 of 32) ² | **58.3** | 4.2 | 0.0 | 0.0 | 15.6% | +0.0 | [−9.1, +9.1] | 1.00 | **401 (1.29×)** | 401 |

¹ Log-polar was measured in the 2026-07-20 `reproduction_eager` campaign, whose
baseline this run reproduces **exactly on every task** (4/6/2/3 successes and
identical grasp rates), so the two are directly comparable. Its per-episode JSON
is in `RetinaBased/GoogleColab/results_reproduction_eager/openvla_foveated/`.

² Not a third point on the same curve — see the adjacency caveat below.

Confidence intervals use a normal approximation on the paired difference and are
**not reliable at small discordant counts**; where they disagree with the exact
test (action repeat 2: CI excludes zero, p=0.057) trust the exact test.

## The baseline is stable, and the three old campaigns are explained

Our baseline reproduces the July `reproduction_eager` campaign task for task.
The three historical OpenVLA/Bridge baselines differ for a mundane reason:

| campaign | baseline | cause |
|---|---|---|
| `results` | 3.1% | earliest run, predates the overlay fix |
| `results_reproduction_sdpa` | 7.3% | **SDPA** attention |
| `results_reproduction_eager` | **15.6%** | **eager** attention |
| this run (2026-08-05) | **15.6%** | eager |

**The attention implementation is worth 8 points.** The Colab setup patch that
forces `attn_implementation="eager"` is load-bearing, not cosmetic. Unlike the
UniVLA baseline dispute, nothing here is unexplained.

## 1. Action repeat: monotone decline, and it is conclusive at k=4

| horizon (env steps/call) | success | Δ | p | speedup |
|---|---|---|---|---|
| 1 (original) | 15.6% | — | — | 1.0× |
| 2 | 7.3% | −8.3 | 0.057 | 2.0× |
| **4** | **4.2%** | **−11.5** | **0.0010** | **4.0×** |

At k=4 the 11 discordant pairs are **11–0**: the condition broke eleven baseline
successes and fixed none. This survives a Bonferroni threshold of 0.05/15≈0.003
across the whole campaign; k=2 alone did not.

This was predicted before the run. OpenVLA emits one action per forward, so it
was trained at horizon 1; every step of repeat moves it further from that, and
the decline is monotone. The contrast with SpatialVLA is the point:

| horizon | OpenVLA | SpatialVLA |
|---|---|---|
| 1 | 15.6% | 32.3% |
| 2 | **7.3%** ↓ | **42.7%** ↑ |
| 4 | **4.2%** ↓ | not yet run |

**Same code, same hook, same direction of horizon change, opposite sign.** The
hypothesis in `UniVLA_Bridge_ActionRepeat.md` — that damage tracks distance from
the horizon the policy was *trained* to execute — predicts SpatialVLA should
*peak* near k=4 (its checkpoint's chunk length) and fall off after. That run is
the cheapest way to test it.

## 2. Depth pruning: the average is flat because two effects cancel

Aggregate success does not move at any depth (15.6 → 17.7 → 16.7 → 15.6, all
p≥0.80). Per task it moves a great deal, and **every task is monotone**:

| task | 0 | 1 | 4 | 8 | direction |
|---|---|---|---|---|---|
| **eggplant** | 25.0 | 41.7 | 45.8 | **58.3** | **strictly up** |
| carrot | 16.7 | 12.5 | 8.3 | **4.2** | strictly down |
| stack | 12.5 | 12.5 | 12.5 | **0.0** | down |
| spoon | 8.3 | 4.2 | 0.0 | 0.0 | down |

One task up, three down, cancelling almost exactly in the mean.

Taken alone, eggplant at 8 layers is 6/24 → 14/24, paired p=**0.0215**. It is one
of four tasks examined, so against a Bonferroni threshold of 0.05/4=0.0125 it
does **not** clear significance. **The evidence is the monotone pattern across a
pre-specified sequence of depths, not this single pairwise p-value.**

### Mechanism: depth is traded for decisiveness, not accuracy

| | baseline | 8 layers bypassed |
|---|---|---|
| eggplant avg steps | **108 / 120** (usually times out) | **77** |
| eggplant grasp → success | 54.2 → 25.0 | 70.8 → **58.3** |
| spoon grasp → success | 12.5 → 8.3 | **33.3** → **0.0** |

Eggplant's baseline failure is *indecision* — it burns the whole 120-step budget
without finishing. Removing layers makes it commit, and it finishes in 77.

Spoon is the mirror image: bypassing layers makes it **grasp far more often**
(12.5% → 33.3%) while success falls to **zero**. It reaches and closes on the
object and then cannot place it.

> **Fixed depth pruning trades precision for decisiveness.** Tasks that were
> failing from indecision improve; tasks that were failing from imprecision
> collapse. An aggregate success rate cannot see this at all.

At 8 layers the pure-inference speedup is **1.29×**, close to UniVLA's static-8
(~1.25×) — so the cost side of the trade is real and comparable across backbones.

### Matched comparison with UniVLA — the same task goes opposite ways

Same ranking rule (`adaptive_sparse_vla/depth_prune.py`), same counts:

| eggplant | 0 | 4 | 8 |
|---|---|---|---|
| UniVLA (Emu3, 32 layers) | 100% | 100% | **88%** ↓ |
| OpenVLA (Llama-7B, 32 layers) | 25.0% | 45.8% | **58.3%** ↑ |

| average | 0 | 4 | 8 |
|---|---|---|---|
| UniVLA | 74% | **81%** (+7) | 78% (+4) |
| OpenVLA | 15.6% | 16.7% (+1.0) | 15.6% (+0.0) |

### Adjacency caveat at 8 layers

Only the back half is eligible (layers 16–31, 16 candidates) and bypassed layers
must not be adjacent, so 8 is exactly the ceiling. The greedy fills the
gap-respecting picks first and then appends the rest, so at 8 the constraint is
relaxed and the selection also varies by task:

| task | bypassed | adjacent? |
|---|---|---|
| eggplant, spoon | 17, 20, 23, **25, 26, 27**, 29, 31 | yes |
| carrot | 17, **19, 20**, 23, 25, 27, 29, 31 | yes |
| stack | 17, **20, 21**, 23, **26, 27, 28**, 30 | yes |

1 layer (`[23]` on every task) and 4 layers (`[17,20,23,27]`, stack
`[17,20,23,26]`) are clean. **Report 8 with this footnote or not at all.**

## 3. Log-polar and blur are not interchangeable

The averages are 34.4% and 33.3%, which earlier writeups summarised as "either
variant works on OpenVLA". Per task they are almost complementary:

| task | log-polar | blur |
|---|---|---|
| eggplant | 33.3 | **62.5** |
| stack | **45.8** | 20.8 |
| spoon | **41.7** | 25.0 |
| carrot | 16.7 | 25.0 |

Blur is far better on eggplant; log-polar is far better on stack and spoon.
**Equal averages do not make them the same intervention**, and the claim in
`LabMeeting_4Backbone_Summary.md` §3.2 that the variant does not matter for
OpenVLA holds only at the aggregate.

## What is established

| result | test | status |
|---|---|---|
| UniVLA/Bridge action repeat, −70.8 | p=1.2×10⁻¹⁹ | **conclusive** |
| UniVLA/LIBERO action repeat, −68.0 | z=−9.8 | **conclusive** |
| OpenVLA foveation log-polar, +18.8 | p=0.0051 | **conclusive** |
| OpenVLA foveation blur, +17.7 | p=0.0060 | **conclusive** |
| **OpenVLA action repeat 4, −11.5** | **p=0.0010** | **conclusive** |
| OpenVLA action repeat 2, −8.3 (and −8.0 on LIBERO) | p=0.057 | suggestive, replicated |
| OpenVLA depth, per-task monotone split | eggplant p=0.0215 (α=0.0125) | pattern, not a single test |
| OpenVLA depth, aggregate | p≥0.80 at every depth | no effect detected |
| **SpatialVLA repeat 2 → repeat 4, −25.0** | **p=3.9×10⁻⁵** | **conclusive** |
| SpatialVLA action repeat 2, +12.5 | p=0.0428 | suggestive |
| SpatialVLA action repeat 4, −12.5 | p=0.0501 | not established |

Across ~15 comparisons a Bonferroni threshold is α≈0.003; the six marked
conclusive clear it. "Not established" means *not detected* — depth pruning at 4
layers has a CI of [−8.3, +10.4] and is compatible with a 10-point gain or an
8-point loss.

The SpatialVLA rows replace the old unpaired +10.4, which was re-measured on
2026-08-05 with per-episode records retained (`SpatialVLA_Bridge_Rerun_0805.md`).
The re-run reproduced the old repeat-2 number on all four tasks. Its two
baseline-relative comparisons straddle 0.05 in opposite directions and neither
is conclusive on its own; the conclusive statement is the *shape* — 30.2 → 42.7
→ 17.7 at horizons 1, 2, 4, with the 2→4 collapse at p=3.9×10⁻⁵. Against
OpenVLA's monotone 15.6 → 7.3 → 4.2 on the same x-axis, the sign of the
intervention differs by backbone.

## Reproduce

```bash
python RetinaBased/PythonProject/simple_eval.py --task <task> --n-episodes 24 \
  --openvla-model-path openvla/openvla-7b --openvla-unnorm-key bridge_orig \
  --device cuda --model openvla                          # original policy
  #  --model openvla_foveated      --foveated-keep-percent 20   # log-polar
  #  --model openvla_foveated_blur --foveated-keep-percent 20   # blur
  #  --model openvla_chunk --action-repeat 2 | 4
  #  --model openvla_depth --depth-prune 1 | 4 | 8
```

`openvla_chunk` is action repeat, not chunk execution — OpenVLA emits one action
per forward, so there is no predicted chunk to truncate. The class is
`ActionRepeatOpenVLAInference`; the name is historical.

Per-episode JSONs: `results/openvla_bridge_0805/`.
