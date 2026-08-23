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


---

## 5. The rule, since "which folder" is the wrong question

Asked 2026-08-22. The folder is not the criterion. **An episode belongs in the
paper's total when a claim in the paper rests on it.** Three candidate rules,
each counted from the files:

| rule | episodes | what it is |
|---|---:|---|
| **1. What Fig. 1 shows** | **4,464** | the grid, five cells at eight conditions |
| **2. What this paper's claims rest on** | **6,670** | rule 1 plus the runs each result needs |
| 3. Every file under `results/` | 7,198 | rule 2 plus a different study and a probe |

### Rule 2 in full, and why each row is in

| episodes | which claim needs it |
|---:|---|
| 4,464 | Fig. 1, result 3's sign reversal, result 2's foveation cells |
| 867 | **result 1.** The window controls *are* the $45.9$-point result, not scaffolding around it |
| 750 | **result 1.** `depth_prune4_early`, `prune4_mid`, `prune4_back`, `prune8`, `prune2_repeat2`, the rest of the layer-choice evidence |
| 288 | **result 2.** The keep sweep that shows the gain peaks at $100\%$ |
| 301 | **contribution 4.** The determinism re-runs the procedure promises |
| **6,670** | |

### Why rule 3 is wrong

**384 episodes are LatentSaccade**, a different intervention. Their files say
so: `model: SpatialVLA+LatentSaccade[OFF (baseline)]`. **240 are a `move_near`
mechanism probe** that no sentence in the Introduction uses. Counting them
inflates the paper by 624 episodes of work it does not report.

`Report.md` line 2201 quotes 7,198 as a file-integrity figure, "255 result
files and 7,198 episodes recounted and correct". That is the right use of it.
It is a statement about the archive, not about the campaign.

### The answer to "should the layer-selection runs count"

**Yes.** They are result 1. The paper's headline claim is that the eligibility
window moves a result by $45.9$ points, and the runs that establish it are the
window controls and the depth variants. They are evidence, not exploration.

### Where each number goes

| number | section | why |
|---|---|---|
| **4,464** | Introduction | it is what Fig. 1 shows, and a reader can derive it from figures already in the paragraph, $3\times96\times8 + 2\times135\times8$ |
| **6,670** | Setup, with the table above | the honest size of the campaign, but it needs the breakdown to be meaningful, and the Introduction has no room for that |
| 7,198 | nowhere in the paper | keep it in `verify_all.py`, where it checks the archive |
