# Mentor results, received 2026-09-11

Unpacked from the mentor's shared folder (data.zip, 35 MB, 1724 files; 301
zero-byte `.task_*.lock` files dropped). Two parts:

- `results/<backbone>_<env>/<configuration>/[<task or suite>/]episodes.jsonl`
  and `summary.json`: one record per episode (seed, success, steps_executed,
  policy_calls, reuses, episode_elapsed_ms, per-episode latency stats) and the
  per-run aggregate with the checkpoint manifest. SimplerEnv WidowX
  (`*_bridge`, `openvla_simplerenv`, `minivla_simplerenv`), Google Robot
  Fractal (`*_fractal`) and LIBERO (`*_libero`, one folder per suite).
- `summarization/{simpler_widowx,google_robot_fractal,libero}/*.csv`: the
  per-backbone CSVs and `summary.csv` (best variant per family) that feed
  Tables I and II. These supersede `artifacts/results/mentor_csv/` (the
  2026-09-03 export). Against that export only the CronusVLA depth-pruning
  rows changed (rerun with the constrained selector on the DiT action
  decoder: blocks 10; 8,10; 4,6,8,10 on WidowX and 3,6,8,10 on Fractal) and
  the MiniVLA motion-entropy row gained its keyframe interval 3 and reuse
  fraction 0.5; empty setting cells now read "N/A". The CronusVLA temporal
  fusion rows are unchanged and still identical across the three settings at
  the episode level.

Checked on arrival: every CSV row (140 SimplerEnv, 167 LIBERO) equals the
aggregate of its episodes.jsonl (successes, episodes, avg_steps,
avg_policy_calls, avg_episode_time_s). Episode sets are identical across all
14 configurations of every backbone and environment, with seed = 42 +
episode_index (CronusVLA records no seed). Exception:
openvla_libero/temporal_fusion_conservative_adaptive lacks the libero_10
suite (300 episodes instead of 400).

Full audit of every file (2026-09-11): experiments/paper/DataAudit_2026-09-11.md,
scripts and lens reports in experiments/data_audit_2026-09-11/. Two points to
know before using the CSVs: the summary.csv files do not all follow the
mentor's selection rule (simpler_widowx picks CronusVLA depth_pruning1 at
35.5 % over depth_pruning2 at 36.0 %; libero breaks eight ties by lower cycle
latency instead of fewer steps), so the table generators select from the
per-backbone CSVs instead; and univla_simplerenv_bridge/temporal_fusion_task_aware
has 81 episodes that crashed with CUDA out of memory and are counted as
failures (48.0 % in the CSV).

Paths inside `summary.json` are the mentor's machine paths and are kept as
received.
