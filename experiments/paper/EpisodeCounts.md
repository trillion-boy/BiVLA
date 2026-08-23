# What 7,198 counts, and why the Introduction now says 4,464

Written 2026-08-22 after the author asked how the episode total is computed
and what "re-runs" re-runs. Both questions found a defect.

## 1. The Introduction used to say the wrong thing

> With its control runs, sweeps and re-runs, the campaign comes to $7{,}198$
> episodes.

$7{,}198$ is what `verify_all.py` counts, and it is correct as a count: every
episode in every one of the 255 result files under `results/`. It is **not**
"the campaign with its controls, sweeps and re-runs", because 624 of those
episodes are neither.

## 2. The full decomposition, counted from the files

| episodes | what it is |
|---:|---|
| **4,368** | the grid, as stored under `results/` |
| **+96** | the OpenVLA/Bridge foveation cell, imported from `RetinaBased/GoogleColab/results_reproduction_eager/` by `build_grid_report.py` |
| **= 4,464** | **the grid, complete** |
| 867 | eligibility-window controls (`*_depth_control`) |
| 288 | the foveation keep sweep (`openvla_bridge_foveate_sweep`) |
| 205 | determinism re-runs |
| 846 | extra conditions some cells ran (`depth_prune8`, `prune4_back`, `prune4_early`, `prune4_mid`, `prune2_repeat2`, `action_repeat2_l4`) |
| **384** | **LatentSaccade, a different study.** `vanilla_baselines/*` and `*_shared_compact_focus/*`, whose files carry `model: SpatialVLA+LatentSaccade[OFF (baseline)]` |
| **240** | **a `move_near` mechanism probe.** `spatialvla_mech_0811` and `spatialvla_move_near_v1_0807`, baseline against `depth_prune4` on one task |
| **7,198** | total under `results/`, minus the 96 imported cell, which lives elsewhere |

The last two rows are the problem. Calling 384 episodes of a **different
intervention** part of "the campaign" overstates the work this paper did, and
no reader could reconstruct the sentence from the files.

## 3. What the Introduction says now

The grid is arithmetic anyone can repeat from figures already in the
paragraph. Three Bridge cells at 96 episodes and two Fractal cells at 135,
eight conditions each:

$$3 \times 96 \times 8 + 2 \times 135 \times 8 = 2304 + 2160 = 4464$$

So both sentences now report **4,464** and describe the rest by what it is
rather than by a total.

- Setup paragraph: *"five of the six cells, each with eight conditions, for
  4,464 episodes. Control runs and sweeps beyond the grid support the results
  below."*
- Contribution 1: *"...for 4,464 episodes. We release the per-episode records
  for the grid and for every control run, sweep and re-run behind the results
  below."*

`verify_all.py`'s `n_ep == 7198` check stays. It is a file-integrity check over
`results/` and it is still right about that. It was never a claim about the
campaign's size, and the paper should not have borrowed it as one.

## 4. What "re-run" means, since it was ambiguous

**The same condition executed a second time, to confirm the simulator returns
the same outcomes.** All 205 episodes:

| run | episodes | what was repeated |
|---|---:|---|
| `spatialvla_fractal_0806/baseline_rerun` | 85 | the SpatialVLA/Fractal baseline. `verify_all` checks all 85 shared episodes match outcome for outcome |
| `univla_bridge_0805/baseline_l4` | 96 | the UniVLA/Bridge baseline, run again on the other card. 11 of 96 episodes differ, which is the 3.1-point bound `Report.md` §3.4.0 quotes |
| `univla_recheck_0810/foveate_blur` | 24 | one task of UniVLA foveation blur. All 24 match |

So yes, identical conditions on identical initial states. Nothing is averaged
across a run and its re-run. The re-runs exist to answer "would we get this
again", and contribution 4's *"check determinism explicitly"* is this.

`univla_bridge_0805/action_repeat2_l4` is a fourth re-run of the same kind, 96
episodes, filed above under extra conditions rather than under re-runs.
