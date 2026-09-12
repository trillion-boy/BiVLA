# Data audit of the mentor's 2026-09-11 export

Every file under `artifacts/results/mentor_2026-09-11/` was opened and
checked: 699 `summary.json`, 699 `episodes.jsonl` (47,500 episode records,
0 malformed lines, 0 duplicate episode keys) and 25 CSVs. The shared folder
held 2,596 entries, which are 1,724 files and 872 directories; 301 of the
files were zero-byte `.task_*.lock` files and were dropped before the
commit, and the 1,423 remaining data files are the ones audited here.
Four independent read-only passes were run (harness settings in
`summary.json`; the CSVs; per-task and per-episode structure; three-way
consistency and the Methods claims), their scripts and raw reports are in
`experiments/data_audit_2026-09-11/`, and the headline numbers below
(errors, step caps, GPU tags, the CronusVLA depth rows) were re-derived
directly from the files before being written here.

Sections: 1 layout, 2 checks that pass, 3 findings that change what the
paper can say, 4 findings that shape the Results text, 5 metadata and
hygiene, 6 what this touches in our text and tables, 7 questions for the
mentor, 8 reproduction.

## Update 2026-09-12 (mentor's report.xlsx, artifacts/results/mentor_2026-09-12/)

- F1 resolved: UniVLA WidowX task-aware rerun, eggplant 50/50, stack cube
  34/50, 164/200 = 82.0 %, no errors, on an RTX PRO 6000 (originals on an
  RTX 5090).
- F2 reuse resolved: CronusVLA WidowX moderate 71/200 (0.14 reuses per
  episode) and aggressive 72/200 (0.935) with the correct thresholds, RTX PRO 6000.
- F2 fusion open: CronusVLA temporal fusion is not in the report.
- F4 confirmed: the mentor marks CogACT task-aware "VALID AS DUPLICATE ONLY".
- Table I cells unchanged. The paired test needs the episodes.jsonl of the
  three repeated runs, not yet received.

## 1. Layout

| backbone and environment | folders | records per `episodes.jsonl` | notes |
|---|---|---|---|
| cogact_base_simplerenv_bridge | 14 (one per configuration) | 200 (4 tasks in one file) | no `inference_cycle_ms` |
| cogact_base_simplerenv_fractal | 70 (14 x 5 tasks) | 50 | |
| cronusvla_simplerenv_bridge / _fractal | 56 / 70 | 50 | minimal schema: no `model_name`, `seed`, `truncated`, latency fields, `fusion` block; carries `temporal_fused_patches` |
| minivla_simplerenv | 14 | 200 (4 tasks in one file) | |
| openvla_simplerenv / _fractal | 56 / 70 | 50 | |
| openvla_libero | 55 | 100 (10 fine tasks x 10) | `temporal_fusion_conservative_adaptive/libero_10` missing |
| spatialvla_simplerenv_bridge / _fractal | 56 / 70 | 50 | `model_name` is `spatialvla_4b_...` |
| univla_simplerenv_bridge / univla_libero | 56 / 56 | 50 / 100 | |
| smolvla_libero | 56 | 100 | two implementations, five GPU types (Section 3, F5) |

Fourteen configurations per backbone and environment: `original`,
`fixed_foveation_keep20/keep50`, `action_repeat2/4`, `depth_pruning1/2/4`,
`guarded_reuse_strict/moderate/aggressive`,
`temporal_fusion_motion_entropy/task_aware/conservative_adaptive`.
Step caps, identical for every backbone on an environment: WidowX 60
(carrot, spoon, stack) and 120 (eggplant), Fractal 80, LIBERO 520 (Long),
300 (Goal), 280 (Object), 220 (Spatial).

## 2. Checks that pass

- `summary.json` aggregates (episodes, successes, success rate, mean steps,
  mean policy calls, mean reuses, mean episode time, every `per_task` field)
  equal the recomputation from `episodes.jsonl` in 699 of 699 folders.
- Every CSV row (140 SimplerEnv, 167 LIBERO) equals the union of its
  folders in episodes, successes, success rate, steps, calls, reuses,
  episode time and the setting columns. All 132 `summary.csv` rows are
  byte-identical to their per-backbone row.
- `seed == 42 + episode_index` in all 41,200 records that carry a seed.
  CronusVLA records none. Episode sets (task, episode index) are identical
  across all 14 configurations of every backbone and environment, except the
  missing OpenVLA LIBERO Long conservative-adaptive run.
- `status` is `completed` in all 699 files. `episode_errors` is 0 wherever
  the field exists (377 files); it is absent in the 322 files of
  openvla_simplerenv, openvla_fractal, spatialvla, univla_bridge and minivla,
  which is why the 81 crashes in F1 are not counted anywhere.
- Depth pruning: a greedy walk in ascending Block Influence that protects
  indices below a quarter of the stack and the final layer and skips
  neighbours of removed layers reproduces the recorded layer set in 138 of
  138 folders that record influence. Sets are nested (budget 1 within 2
  within 4). The leading protection was binding in 33 of 48 distinct
  selections (the lowest-influence layers sit at indices 2 to 6), the gap
  rule in 40, the final-layer protection never (so it is untestable from the
  data). OpenVLA LIBERO Spatial removes index 8 of 32, the first unprotected
  index under the floor rule.
- Action repeat: `policy_calls == ceil(steps / (m k))` in 100 % of the 2,600
  action-repeat episodes, with m = 1 everywhere except UniVLA (m = 5 on
  WidowX, 10 on LIBERO, its native chunk). SpatialVLA has a native chunk of
  4 but the harness queries it every step.
- Foveation: `fovea_keep_ratio` is 0.2 in every keep20 folder and 0.5 in
  every keep50 folder, all 13 backbone-environments.
- Temporal fusion, keyframe interval 3: `fusion.keyframes` equals the sum of
  `ceil(calls / 3)` over episodes exactly in all 76 folders that record it.
- Guarded reuse: `reuses <= cap x calls` holds in all 8,600 guarded-reuse
  episodes; `steps == calls + reuses` in every non-UniVLA guarded episode.
- Every failure that did not crash ends exactly at the task's step cap:
  21,495 non-error failures, 0 that stopped early. Both simulators run a
  failed episode to the horizon.

## 3. Findings that change what the paper can say

F1. UniVLA WidowX task-aware fusion crashed on 81 of 200 episodes. All 81
records carry `torch.OutOfMemoryError: CUDA out of memory` (42 in
put_eggplant_in_basket, 39 in stack_cube; 5 to 55 steps executed; the GPU
had 31.4 GiB with about 30.9 GiB in use). They are stored as failures, so
the CSV row reads 48.0 % against 87.5 % for Original. On the 119 episodes
that did not crash the setting scored 96 of 119 (80.7 %), and on those same
episodes Original scored carrot 42 vs 36, eggplant 7 vs 8, spoon 44 vs 44,
stack 9 vs 8. Table I is not affected because the selection rule picks
conservative-adaptive for that cell, but conservative-adaptive is inert on
UniVLA (F3), so UniVLA WidowX has no valid temporal-fusion result. The
full-sweep paired result of -39.5 points for this setting
(`PairedResultsAll.md`) is an artefact of the crashes, not a model effect.

F2. CronusVLA settings that did not run as named.
- WidowX guarded reuse: all three presets record the strict thresholds
  (frame 0.01, patch 0.03, cosine 0.995, cap 1) and are episode-identical
  (72 of 200 successes each, 5 reuses in 9,947 steps). The Table I cell is
  the strict setting only. Fractal presets are distinct and behave
  distinctly (136, 137, 142 of 250).
- Temporal fusion, both environments: no fusion argument exists in any of
  the 126 CronusVLA files, and the three settings are episode-identical in
  success, steps, calls and `temporal_fused_patches` (WidowX 67 of 200 each,
  Fractal 143 of 250 each); only wall-clock differs. The CSV reports keyframe
  interval 1 and fraction 0.5 for all three rows, which cannot produce the
  102 to 118 fused patches per call the episodes record. The rerun the mentor
  mentioned on 2026-09-11 is not in this export.
- Depth pruning is verified: 12-entry influence vector (the DiT action
  decoder), explicit rules (`min_layer_fraction 0.25`, `protect_last 1`,
  `min_gap 1`), blocks 10; 8,10; 4,6,8,10 on WidowX and 3,6,8,10 on Fractal.
  The 27 depth folders come from a different output directory
  (`..._protected/`) and checkpoint path than the other 44 CronusVLA folders.
- The CronusVLA schema has no seeds, no per-call latency, no `model_name` and
  no `per_task` block. Its CSV policy latency equals the median over episodes
  of `episode_elapsed_ms / policy_calls`, so for CronusVLA `cycle` and
  `policy` are the same number (12 of 12 WidowX rows equal).

F3. Conservative-adaptive fusion never engages on SpatialVLA and UniVLA.
In all 17 such folders `fusion.keyframes == policy_calls` and
`median_reusable_visual_tokens == 0`; UniVLA WidowX is episode-identical to
Original in success, steps and calls (200 of 200), UniVLA LIBERO differs in
8 of 400 episodes, SpatialVLA Fractal in 3 of 250. The rule picks this
setting for the Table I UniVLA WidowX fusion cell (87.50, +0.00) and for the
Table II UniVLA Object and Spatial fusion cells, which are therefore the
Original run under another name. On OpenVLA, CogACT and MiniVLA the setting
does engage (reusable tokens 64 = 0.25 x 256, the cap binding).

F4. CogACT task-aware fusion is motion-entropy under another name.
`fusion.task_relevance_supported` is `false` in all 84 CogACT files, no
`collect_relevance` flag exists, and task-aware is identical to
motion-entropy per file in keyframes, reusable tokens, policy calls and
successes (WidowX 116 of 200 both; all five Fractal tasks). CogACT tested
two fusion settings, not three. On every other backbone that records the
flag, task-aware collects attention and differs from motion-entropy.

F5. SmolVLA is two implementations on five GPUs.
- 27 "legacy" folders (Original, foveation x2, action repeat x2, depth 1 on
  all suites, depth 2 on Object and Spatial, depth 4 on Spatial) record
  transformers 4.51.3 and lerobot 0.4.4; 29 "reconstructed" folders (guarded
  reuse x3 and temporal fusion x3 on all suites, depth 2 on Long and Goal,
  depth 4 on Long, Goal and Object) carry only the tag
  `smolvla-reconstructed-sdpa-v1`, no arguments, no checkpoint manifest, and
  the claim scope "Legacy and new implementation/hardware timings are not a
  matched comparison; no improvement claim".
- Reconstructed episodes carry a `gpu` field: RTX 5090 (most), RTX 6000 Ada,
  RTX PRO 6000 Blackwell, L40S, RTX A6000, mixed inside single folders.
  Query latency is about 262 ms on the RTX 5090, 184 to 189 ms on the L40S
  and 463 to 484 ms on the A6000, against 275 ms for the legacy Original.
  Three folders mix implementations (depth 2 Long 83 + 17, depth 2 Goal
  74 + 26, depth 4 Long 38 + 62).
- Depth pruning on SmolVLA is not the Methods procedure: `calibrated:
  false`, `influence: null`, fixed layers [30], [28,30], [24,26,28,30]
  although the VLM keeps 16 layers, and policy latency does not fall (275.1,
  277.6, 277.3 ms for budgets 0, 1, 2) while success collapses (Long 42, 9,
  3, 0 of 100). What the indices address cannot be read from the files.
- The SmolVLA CSVs report `median_reusable_visual_tokens = 0` for all 12
  fusion rows while the episodes record 16 (conservative), 19 to 26
  (motion-entropy) and 3 to 7 (task-aware).
Consequence: every SmolVLA latency change in Table II for guarded reuse,
temporal fusion, depth 2 on Long and Goal and depth 4 on Long, Goal and
Object compares two implementations on different hardware and is not a
valid delta; the success changes are also not matched runs.

F6. Guarded reuse rarely opens its gates. Reused steps as a share of all
steps, strict / moderate / aggressive (totals from `per_task_success.csv`):
OpenVLA LIBERO 3.6 / 5.2 / 10.1 %, OpenVLA WidowX 3.7 / 4.8 / 6.3 %, OpenVLA
Fractal 2.3 / 2.5 / 3.9 %, MiniVLA 3.1 / 4.2 / 4.0 %, CronusVLA Fractal
0.7 / 1.7 / 4.2 %, SmolVLA 0.1 / 0.4 / 2.3 %, CogACT WidowX 0.3 / 0.9 /
2.7 % (32 / 97 / 271 reuses in about 10,000 steps), SpatialVLA Fractal
0.5 / 0.8 / 2.4 %, CogACT Fractal 1 / 12 / 62 reuses in about 11,000 steps,
SpatialVLA WidowX 2 / 6 / 38 reuses in about 9,000, CronusVLA WidowX 5 / 5 /
5 in 9,947, UniVLA LIBERO 5 / 6 / 17 in about 65,600, UniVLA WidowX 0 / 1 /
2 in about 6,000. The paired result "guarded reuse never significant" is
therefore mostly "the gates almost never opened", and on UniVLA, SpatialVLA
WidowX and CronusVLA WidowX the guarded-reuse rows are the Original run to
within a handful of steps (UniVLA WidowX strict and all three UniVLA LIBERO
Object and Spatial guarded rows equal Original in success, steps and calls
in the CSVs).

F7. The mentor's `summary.csv` files do not all follow the stated rule
(highest success, then fewer steps). `simpler_widowx/summary.csv` carries
CronusVLA depth_pruning1 (35.5 %, 50.03 steps) although depth_pruning2 has
36.0 % (50.52 steps); no rule tested reproduces that pick. `libero/summary.csv`
breaks all eight of its success ties by lower `cycle_median_latency_ms`
rather than fewer steps (OpenVLA Long and Object guarded reuse, SmolVLA Long
fusion and Spatial guarded reuse, UniVLA guarded reuse on all four suites).
`google_robot_fractal/summary.csv` has no deviation. Both table generators
now select from the per-backbone CSVs by the rule and print every cell where
a `summary.csv` differs; Table I CronusVLA WidowX depth pruning is now
36.00 (+2.00) / 118.76 (-7.08) / 50.52 (-0.04).

F8. Missing run: `openvla_libero/temporal_fusion_conservative_adaptive` has
no `libero_10` folder, so `libero/openvla_libero_10.csv` has 13 rows and the
OpenVLA Long fusion cell is chosen from two settings (motion-entropy 49 %,
task-aware 48 %).

## 4. Findings that shape the Results text

R1. The latency columns of the CSVs mean different things per harness. For
CronusVLA `cycle == policy` (no environment time inside the cycle); for
CogACT and SpatialVLA the cycle includes the environment step (40 to 97 ms
more than policy); for OpenVLA, MiniVLA and SmolVLA it adds 2 to 34 ms; for
UniVLA one cycle spans one call, which is one chunk of 4.7 to 9.7
environment steps. Under action repeat the cycle rises for CogACT,
SpatialVLA and UniVLA (per call including the repeated steps), falls to
policy / k for CronusVLA and OpenVLA SimplerEnv (amortised), and falls to
the reuse-step median (2.2 ms for OpenVLA LIBERO k = 4, 7.7 ms MiniVLA,
12 to 19 ms SmolVLA). `control_frequency_hz` has three definitions
(environment steps per second, 1000 / cycle, or policy calls per second)
and rises under action repeat for some backbones and falls for others.
The UniVLA WidowX cycle column cannot be reproduced from the episodes at
all (6 to 16 ms below every candidate). The per-step wall-clock the tables
use, `1000 x avg_episode_time_s / avg_steps`, is the only latency quantity
with one meaning for every harness, and the CSV values of it reproduce
exactly from the episodes.

R2. Latency per family, per-step wall-clock relative to Original (ranges
over settings and environments): foveation is slower everywhere, +2 to
+17 % (the blur runs on the CPU path; CogACT and CronusVLA +12 to +17 %,
OpenVLA and UniVLA under +5 %). Action repeat k = 2 gives 0.53 to 0.73 of the
Original per-step time and k = 4 gives 0.28 to 0.59, never 1 / k, because
environment stepping stays and the per-call time rises by 80 to 220 ms on
CogACT, CronusVLA, OpenVLA, SpatialVLA and UniVLA (the environment is
stepped inside the call). Depth pruning lowers per-call latency
monotonically on every backbone with latency fields except SmolVLA (budget 4
at 0.87 to 0.95 of Original) and is the only non-repeat family that lowers
per-step time for at least one budget on every backbone, by up to 17 %
(SmolVLA Object budget 4, at 13 % success). Guarded reuse changes
per-step time by -13.6 to +4.2 %, mostly within 3 %. Task-aware fusion
raises per-call latency by 20.5 % on UniVLA WidowX, 13.6 % on MiniVLA,
12.8 % on UniVLA LIBERO and 7.5 to 8.0 % on OpenVLA, and not at all on
CogACT (F4) and SpatialVLA (+0.8 to +1.3 %); motion-entropy and
conservative-adaptive stay within 2 % of Original everywhere.

R3. Avg. Steps tracks success. Since every failure runs to the cap, the
mean step count is the success-weighted average of the successful episodes'
lengths and the cap, so a column change in Avg. Steps is mostly a change in
success rate. The `truncated` flag is not usable for this: absent in
CronusVLA and SmolVLA, always false on LIBERO (2,934 failures at cap carry
false), set only for move_near and pick_coke_can on Fractal, and set on 18
successes that ended at the cap. CronusVLA Fractal has 121 successes at the
cap (72 move_near, 47 pick_coke_can); that harness runs those tasks to the
horizon and scores at the end.

R4. Where a drop comes from, per task (successes of 50 unless noted).
- Action repeat collapses on every task: UniVLA WidowX 42, 49, 44, 40 to
  12, 3, 6, 4 at k = 2; UniVLA LIBERO k = 4 puts 38 of the 40 fine tasks
  that had at least 50 % at 0 %; CogACT WidowX stack_cube 16 to 0; CronusVLA
  WidowX all four tasks to 0 at k = 4. On Fractal it is flat or positive at
  k = 2 for all four backbones.
- Depth pruning drops are concentrated: MiniVLA budget 1 is stack_cube 36
  to 8 while eggplant 9 to 10 and spoon 25 to 19; budget 2 and 4 are 0 on
  all four tasks. SpatialVLA WidowX budget 4 is eggplant 50 to 8 with carrot
  unchanged at 13; budget 2 is spoon 11 to 0. SmolVLA depth hits Long
  hardest (42, 9, 3, 0 of 100), and budget 4 leaves 20 of 40 fine tasks at 0
  while Goal task 4 and 7 stay 10 of 10.
- Foveation on OpenVLA is one or two tasks: WidowX eggplant 37 to 8 (keep
  20) and 12 (keep 50) with spoon 19 to 19 and 24; Fractal keep 20 is
  move_near 31 to 3, pick_coke 29 to 1, open_drawer 9 to 0 while close_drawer
  20 to 15; keep 50 halves those drops. LIBERO keep 20 drops all four
  suites, keep 50 is within one point except Long 56 to 44 and Spatial 78
  to 83. Keep 20 is worse than keep 50 on every Fractal and LIBERO run and
  on MiniVLA and OpenVLA WidowX; on WidowX the two tie for CogACT (105 of
  200 each) and keep 20 leads for CronusVLA (73 vs 68), SpatialVLA (100 vs
  89) and UniVLA (158 vs 146).
- Gains that reach significance in the paired test (`PairedResults.md`):
  CogACT WidowX depth 2, 50 to 60 % (p = 0.009); CogACT WidowX motion-entropy
  fusion, 50 to 58 % (p = 0.020); OpenVLA Fractal task-aware fusion, +6.8
  points (p = 0.012); UniVLA Goal foveation keep 50, +7 (p = 0.039). No other
  table cell gains significantly.

R5. Per-episode latency is stable within a run: on CogACT, MiniVLA, OpenVLA
and SpatialVLA the 10th to 90th percentile of per-episode median query
latency spans 0.3 to 1.5 % of the run median, on UniVLA 9 to 17 %; no run
has an episode above 1.64 x its run median, and no run has repeated
identical values. Per-episode Original medians: CogACT 98.7 / 99.0 ms
(WidowX / Fractal), MiniVLA 97.1, OpenVLA 159.0 / 160.1 / 158.5 (LIBERO),
SmolVLA 275.1, SpatialVLA 382.4 / 379.0, UniVLA 714.5 (WidowX) / 473.6
(LIBERO); CronusVLA has no per-call latency and its elapsed per call is
126.3 / 152.6 ms.

R6. Bookkeeping differences that the Results must not read as effects.
CogACT and SpatialVLA record `reuses = 0` under action repeat (calls still
equal `ceil(steps / k)`), the other harnesses record `steps - calls`, and
UniVLA records `(k - 1) x chunk x calls` without horizon truncation (LIBERO
Goal k = 4 reports 240 reuses on 300-step episodes, 320 > 300). `avg_reuses`
in the CSVs inherits these. Every non-repeat, non-reuse episode has 0
reuses (no leakage between families).

## 5. Metadata and hygiene

- Attention backend is recorded only for SpatialVLA (`native_mixed (SDPA
  where supported; ZoeDepth compatible)`, argument `sdpa`) and UniVLA
  (`sdpa`). OpenVLA, CogACT, CronusVLA, MiniVLA and SmolVLA record none. No
  file records a torch version. Library versions exist only for SmolVLA
  legacy (transformers 4.51.3, lerobot 0.4.4). The Setup handoff item
  "OpenVLA LIBERO backend" stays open.
- Run settings that are recorded and consistent: seed 42; 50 trials per
  SimplerEnv task and 10 per LIBERO fine task; `max_steps` 150 on WidowX
  (task caps 60 and 120 apply), 80 on Fractal, 220 / 280 / 300 / 520 on
  LIBERO; `resume: true` on the LIBERO runs; CogACT cfg scale 1.5 and 10
  DDIM steps; SpatialVLA chunk 4 with ensemble temperature -0.8; UniVLA
  chunk 5 (WidowX) and 10 (LIBERO); OpenVLA and MiniVLA 224 px without
  centre crop. `arguments.action_repeat` is 2 and `depth_layers` is 1 on
  every unrelated row, so the configuration name is the only reliable key.
- CSV hygiene: CRLF line endings; success rate written as `50` or `36.0`
  depending on the file; `selected_depth_layers` separated by `;` except
  CronusVLA (`,` and quoted); `libero/summary.csv` drops the `suite` column
  and renames the environment to `Libero Long`, `LIBERO-Goal`,
  `LIBERO-Object`, `LIBERO-Spatial`; `checkpoint_model_id` for UniVLA LIBERO
  is an absolute home-directory path on the mentor's machine (five files);
  `openvla-7b` names both the WidowX and the LIBERO checkpoints; the MiniVLA
  CSV is in alphabetical rather than family order; `median_reusable_visual_tokens`
  is `0` rather than `N/A` in non-fusion rows.
- `univla_libero` claim scope is copied from the WidowX file ("paired
  SimplerEnv WidowX rollouts, native five-action chunks") although the
  LIBERO chunk is 10.

## 6. What this touches in our text and tables

Methods (Section III). Every mechanism sentence holds for the backbones
that record arguments: the depth selector, the keyframe rule, the reuse cap,
the action-repeat bookkeeping and the keep ratios are all reproduced from
the fields (Section 2). Three sentences are true as protocol but not as
executed on every backbone and need a footnote or a rerun before Results
cites them: "Three settings are evaluated" for fusion (CogACT ran two, F4;
CronusVLA ran one unknown setting, F2), "strict, moderate, or aggressive"
for reuse (CronusVLA WidowX ran strict three times, F2), and the depth
selector (SmolVLA used fixed uncalibrated indices, F5). "Protects a leading
fraction of the stack" matches the floor rule the data follows (OpenVLA
LIBERO removes index 8 of 32). The lead-in's "under the attention backend
that Section IV-A lists for its harness" depends on the mentor's Setup,
since the data records a backend for two of seven backbones only.

Related Work (Section II). Nothing to change; the paragraph on matched
episodes and the dense implementation stays accurate.

Introduction (mentor's draft, revised copy in the scratchpad). The sentence
that action repeat passes in one SimplerEnv setup and fails in the other,
and that foveation passes and fails between LIBERO suites, is supported
(R4). Any sentence that presents task-aware fusion as run on every backbone
must except CogACT (F4) and CronusVLA (F2), and any UniVLA WidowX fusion
statement must except the crashed setting (F1).

Table I. CronusVLA WidowX depth pruning updated (F7). The CronusVLA guarded
reuse cell is the strict setting; the CronusVLA fusion cells are one
setting of unknown parameters; the UniVLA WidowX fusion cell is an inert
run (F3). Footnotes or reruns are needed for those three cells.

Table II. Selection by the rule differs from the mentor's summary in eight
cells (F7). SmolVLA latency changes for the reconstructed rows are not
matched (F5). UniVLA Object and Spatial fusion cells are inert runs (F3);
UniVLA Object and Spatial guarded-reuse cells are byte-identical to
Original (F6). OpenVLA Long fusion is chosen from two settings (F8).

Results (Section IV-B and IV-C, not yet written). Use `PairedResults.md`
for the table cells and `PairedResultsAll.md` for the sweep, exclude or
mark the UniVLA WidowX task-aware row (F1), report gate-firing rates next to
the guarded-reuse verdict (F6), describe latency as per-step wall-clock and
say why (R1), and state that Avg. Steps follows success because failures run
to the cap (R3).

## 7. Questions for the mentor

The mentor's second reply (2026-09-11) settled the tie rule, the parameter
counts, the depth-pruning mechanism, the blur foveation and the
conservative-adaptive zeros ("retain the value and flag the configuration
as ineffective"). What remains, split by round.

Sent in the current reply (a rerun or a confirmation is needed):

1. UniVLA WidowX task-aware: 81 of 200 episodes crashed with CUDA out of
   memory and are counted as failures (48.0 %). Rerun on a larger GPU, or
   mark the setting as not run for UniVLA WidowX?
2. CronusVLA WidowX guarded reuse: the moderate and aggressive folders
   record the strict thresholds and give identical episodes. Rerun of the
   two settings?
3. CogACT task-aware: `task_relevance_supported` is false and the setting
   equals motion-entropy episode for episode. No rerun needed; confirm the
   footnote "CogACT has two fusion settings rather than three".

Held for the next round:

4. CronusVLA temporal fusion: no fusion arguments are recorded and the
   three settings are episode-identical on both environments; the CSV says
   keyframe interval 1. The mentor reported a rerun and shared logs, but the
   export does not contain it. Which interval and fraction ran, and is the
   rerun still coming?
5. SmolVLA: (a) the 29 reconstructed folders are a different implementation
   on five GPU types, so latency deltas against the legacy Original are not
   matched. Rerun on one GPU and one implementation, or drop SmolVLA latency
   from Table II? (b) The depth layers 30; 28,30; 24,26,28,30 exceed the
   16-layer VLM and are marked uncalibrated. What do they index?
6. OpenVLA LIBERO Long conservative-adaptive is missing. Will it be run?
7. Selection (low priority, a notice rather than a question): the nine
   cells where the mentor's summary.csv departs from the rule were reselected
   by the rule.
8. Setup (the mentor's section): the attention backend is recorded only for
   SpatialVLA and UniVLA; no torch version is recorded; the GPU is recorded
   only in the SmolVLA reconstructed runs.

Writable before any answer arrives: IV-B, IV-C, V, the abstract and the
introduction, with the three affected cells left as footnote slots; the
answers change those cells and the presence of footnotes, not the
conclusions (two significant gains, control collapse, inert reuse).

## 8. Reproduction

- `experiments/data_audit_2026-09-11/`: scripts, `per_task_success.csv`
  (2,286 rows), `summary_json_index.csv` (699 rows, 101 columns), the raw
  outputs and the four lens reports.
- `experiments/paired_analysis.py` writes `paper/paired_results.csv`,
  `PairedResults.md`, `paired_results_all.csv`, `PairedResultsAll.md` and
  stops unless all 699 `episodes.jsonl` were read.
- `experiments/make_simpler_table.py` and `make_libero_table.py` select by
  the rule from the per-backbone CSVs and print every deviation from the
  mentor's `summary.csv`; `render_table_preview.py` draws Table I.
