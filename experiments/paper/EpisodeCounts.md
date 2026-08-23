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

## 4. What "re-run" means, and what it actually found

⚠️ **Corrected 2026-08-22.** An earlier version of this section filed
`baseline_l4` and `action_repeat2_l4` as diagnostic re-runs. They are the
opposite. `build_grid_report.py` lines 146--153 remap them:

```python
RENAME = {("UniVLA", "Bridge"): {
    "baseline": None, "action_repeat2": None,        # the July card, dropped
    "baseline_l4": "baseline",                        # the grid's baseline
    "action_repeat2_l4": "action_repeat2"}}
```

**The L4 pair is the grid. The July pair is the diagnostic.** The comment
above that map gives the reason: UniVLA/Bridge was measured on two cards, and
subtracting across that boundary would put a hardware component inside every
delta, so the column was rebuilt around the L4 runs and the July runs dropped
out of the grid without being deleted.

### The three repeats, recomputed

| pair | shared episodes | differ |
|---|---:|---:|
| SpatialVLA/Fractal `baseline` vs `baseline_rerun`, same environment | 85 | **0** |
| UniVLA/Bridge `foveate_blur` vs `recheck_0810`, same L4 | 24 | **0** |
| UniVLA/Bridge July `baseline` vs `baseline_l4`, **different card** | 96 | **11** |

### So the conclusion is not "it is deterministic"

Repeat a condition **in the same environment** and every episode reproduces,
85 of 85 and 24 of 24. Move to a different card and **11 of 96 episodes
flip**, which is the 3.1-point gap `Report.md` §3.4.0 quotes.

That second row is why the repeats exist and what they changed. They did not
end in "fine, it is deterministic, carry on." They found a boundary the grid
could not subtract across, and the UniVLA/Bridge column was **restructured**
so that every delta in it is computed within one card. Contribution 4's
*"check determinism explicitly"* is this, and the paper's rule that a
condition and its baseline must come from the same environment comes from
here rather than from an assumption.

`Hardware.md` §3 carries the related finding, that the policy itself is
numerically stable across cards while the **foveation image path** is not,
which points at a `cv2` build difference rather than at the GPU.

## 5. The rule, and the number the paper reports

⚠️ **Corrected 2026-08-22, second time.** This section first recommended
**4,464**, the grid alone. That was wrong, and the author caught it. Result 1's
losing arm, `window875` at $-30.4$, is a **control run and sits outside the
grid**, so a reader who tried to locate the $45.9$-point headline result inside
4,464 episodes could not. Reporting a total that excludes the evidence for the
paper's biggest claim is worse than reporting nothing.

### The rule

The folder is not the criterion. **An episode belongs in the paper's total when
a claim in the paper rests on it.** `Report.md` line 585 already draws exactly
the right line, between the **grid** and the **control and diagnostic runs**,
and puts `baseline_rerun`, `depth_prune4_early`, `_mid`, `move_near_v1` and the
determinism re-runs in the second column. Both columns are this campaign. What
is *not* this campaign is a different intervention.

### The arithmetic

| | episodes |
|---|---:|
| every file under `results/` | 7,198 |
| minus LatentSaccade, a different study | $-384$ |
| plus the OpenVLA/Bridge foveation cell, imported from `RetinaBased/` | $+96$ |
| **this campaign** | **6,910** |

which splits as:

| bucket | episodes | which claim needs it |
|---|---:|---|
| the grid | 4,464 | Fig. 1, result 3, result 2's foveation cells |
| eligibility-window controls | 867 | **result 1.** `window875` and friends are one arm of the $45.9$-point contrast |
| extra depth variants | 750 | **result 1.** `prune4_early`, `_mid`, `_back`, `prune8`, `prune2_repeat2` |
| the keep sweep | 288 | **result 2.** The four-point curve peaking at $100\%$ |
| same-environment re-runs | 109 | **contribution 4.** 85 of 85 and 24 of 24 reproduce |
| the July-vs-L4 pair | 192 | **contribution 4.** 11 of 96 differ, the 3.1-point hardware bound |
| `move_near` mechanism and task-version probe | 240 | the failure-typing section |
| **campaign** | **6,910** | |

### Why the 384 stay out

Their files say what they are: `model: SpatialVLA+LatentSaccade[OFF
(baseline)]`. LatentSaccade is a different intervention, from the
`RetinaBased` line of work, and no sentence in this paper reports it. Counting
it would inflate the campaign by a study the paper does not describe.

### What the sections say

| number | where | why |
|---|---|---|
| **6,910** | Introduction, both places | the campaign, including the runs result 1 needs |
| 4,464 and the split above | Setup | the grid against its controls, which needs the table to mean anything |
| 7,198 | nowhere in the paper | it stays in `verify_all.py` as an archive check, which is how `Report.md` line 2201 uses it |

### The answer to "should the layer-selection runs count"

**Yes, and that is the whole point.** They are result 1, not exploration around
it. The headline claim is that the eligibility window moves a result by $45.9$
points, and the window controls are one of the two arms of that contrast.


---

## 6. The date rule, applied 2026-08-22

The author's rule: **this paper's simulations are the late-July and August
runs, and where the same run exists twice, the later one counts.** Applied by
reading the date each file entered git, not the filesystem mtime, which is
clone time in this container.

### Every campaign directory by date

| entered git | episodes | directory | verdict |
|---|---:|---|---|
| **2026-06-18** | 96 | `spatialvla_shared_compact_focus` | **out.** June |
| **2026-06-18** | 96 | `univla_shared_compact_focus` | **out.** June |
| **2026-06-18** | 192 | `vanilla_baselines` | **out.** June |
| 2026-07-13 | 96 | the imported OpenVLA/Bridge foveation cell | **in**, see below |
| 2026-08-05 to 08-12 | 6,814 | everything else under `results/` | in |
| | **6,910** | **campaign** | |

**The date rule and the different-study rule give the same answer.** The 384
episodes the June rows hold are exactly the LatentSaccade files, identified
independently by their `model: SpatialVLA+LatentSaccade[OFF (baseline)]`
field. Two criteria, one answer, so 6,910 stands.

### Same run twice, later one wins

| pair | dates | which counts |
|---|---|---|
| UniVLA/Bridge `baseline` vs `baseline_l4` | 08-05, **08-10** | the **08-10** L4 run, which `RENAME` already installs as the grid's baseline. The rule and the code agree |
| UniVLA/Bridge `action_repeat2` vs `_l4` | 08-05, **08-10** | the **08-10** run, same mechanism |
| SpatialVLA/Fractal `baseline` vs `baseline_rerun` | **08-06** (135 eps), 08-07 (85 eps) | the **08-06** run. The later file is 85 of 135 episodes, a partial re-check rather than a replacement, so "later wins" does not apply |

The two superseded 08-05 runs, 192 episodes, stay inside the 6,910. They are
no longer grid cells, but the 3.1-point hardware bound is the comparison
between them and the 08-10 pair, so a claim rests on them.

### The one July file, and why it is admitted

`RetinaBased/GoogleColab/results_reproduction_eager/` entered git on
**2026-07-13** and holds 288 episodes in three directories. Only 96 enter the
count, `openvla_foveated`, the OpenVLA/Bridge foveation cell. The other two
are that campaign's own baseline and an unused `openvla_retina` run.

The `openvla_retina` directory is **not** counted and not used anywhere. The
`openvla` directory is not counted either. It exists only so the import can be
checked, because `build_grid_report.py` admits the borrowed condition solely
when that campaign's own baseline matches the grid's baseline episode for
episode. Recomputed today:

| | episodes | differ |
|---|---:|---:|
| their July baseline against our August grid baseline | 96 shared | **0** |

**A July run and an August run of the same condition agree on all 96
episodes.** So the July date is not a cross-session risk here, and this is a
fourth determinism data point alongside the 85 of 85, the 24 of 24 and the
11 of 96 across cards.
