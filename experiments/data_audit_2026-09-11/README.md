# Audit scripts and outputs for the 2026-09-11 export

Read-only checks over every file in `artifacts/results/mentor_2026-09-11/`
(699 `summary.json`, 699 `episodes.jsonl` with 47,500 episode records, 25
CSVs). The consolidated findings are in `experiments/paper/DataAudit_2026-09-11.md`.
The four lens reports below are the raw reports the checks produced; the
consolidated document is the one to read.

Scripts (absolute paths to the repository at `/home/user/BiVLA`; edit the
path constants at the top to run elsewhere):

- `build_index.py`: flattens every `summary.json` into `summary_json_index.csv`
  (699 rows, 101 columns: status, counts, every argument, checkpoint manifest,
  fusion and depth calibration fields).
- `load.py`, `analyze.py`, `followup.py`: load every `episodes.jsonl`, write
  `per_task_success.csv` (2,286 rows, one per backbone, configuration and task
  or LIBERO fine task) and the per-task, truncation, error, latency, reuse and
  seed checks (`followup_out.txt`).
- `audit.py`: the 25 CSVs, structure, column semantics per harness, the
  selection rule of each `summary.csv`, and latency per step (`audit_out.txt`).
- `verify.py`, `verify2.py`, `verify3.py`: `summary.json` against
  `episodes.jsonl`, CSV rows against the union of their folders, and the
  Methods claims (depth selector, fusion keyframes, reuse cap, action repeat
  bookkeeping, foveation keep ratio) against the recorded fields (`report2.txt`).

`file_manifest.csv`: every one of the 1,724 files in the zip (path, kind, bytes,
sha256, parse result); the 301 lock files are all zero bytes and the 1,423
data files are byte-identical to the committed copy.

Lens reports: `lens_settings.md` (harness settings in `summary.json`),
`lens_csv.md` (the CSVs), `lens_episodes.md` (per-task and per-episode
structure), `lens_consistency.md` (three-way consistency and Methods claims).
