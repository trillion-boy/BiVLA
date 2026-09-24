# Project page: notes for the authors (not part of the published page)

## What is here

- Upload only the contents of `site/` (or unzip `project_page.zip`). Nothing else in this folder (this file, `content/`, `data/`, `website_data/`) may go into the anonymised repository.
- `site/` is the complete static page: `index.html`, `assets/` (two videos, ten wall clips with posters), `figs/` (figures). No scripts, no external requests, no fonts loaded from the network. Copy the **contents of `site/`** to the root of the anonymised repository behind `https://anonymous.4open.science/w/icra-bot/` so that `index.html` sits at the root; the `/w/` link then renders it. Total size about 23 MB (videos 15 MB, wall clips 8.6 MB, figures 3.9 MB; only the figures the page uses are copied).
- `build.py`, `style.css`, `content/*.md`, `data/*.csv`, `figures.json`, `walls.json`, `figs/`, `media/` are the sources. `python3 build.py` regenerates `site/index.html` (needs the `markdown` package).
- `data/full_settings.csv` (308 rows) and `data/per_task.csv` (2,296 rows) are the machine-readable tables behind Sections 2.1 and 2.2. They carry no paths or names and could be published as well; the raw `summary.json` / `episodes.jsonl` files must **not** be published as they are, because their `arguments`, `checkpoint_manifest`, `output_dir`, `slurm_job_id` and `qos` fields contain user names and cluster identifiers.

## Statements on the page that the authors should confirm before publishing

The final paper promises on the project page "checkpoints, GPU cards, software versions, trick presets, per-cell p-values, and implementation caveats". The page delivers all six, and the following items go beyond what the paper states. Each is true of the run records; the question is only whether the authors want it public.

1. **GPU cards** (Section 1.3). The paper says "a system equipped with 4x RTX 5090 GPUs". The 15 and 16 September reruns (SmolVLA reuse and fusion, CronusVLA WidowX moderate and aggressive reuse, UniVLA WidowX task-aware fusion) all record an RTX 5090. The only records that still list other cards are 86 of the 500 episodes of the SmolVLA depth-pruning cells at two layers (Long, Goal) and four layers (Long, Goal, Object), dated 6 and 7 September (RTX PRO 6000, RTX 6000 Ada, L40S, RTX A6000). The page states this in one sentence. If those cells were also rerun on the RTX 5090, send the new files and the sentence goes away.
2. **Two implementations of SmolVLA** (Section 1.6 and 3.13). Table II prints latency for the SmolVLA reuse, fusion and depth cells; the page says those cells mix an implementation change with the trick and should be read through the paired test alone. This is the paper draft's own caveat (setup.tex, 15 Sept) that did not survive into the final text.
3. **CogACT ran two fusion settings, not three** (task-aware equals motion-entropy because CogACT exposes no text-to-vision attention where fusion runs). The paper counts thirteen settings for every backbone.
4. **Inert cells**: conservative-adaptive fusion never engaged on SpatialVLA and UniVLA; ten UniVLA reuse cells fired no gate; CronusVLA's three fusion settings are one run with unrecorded parameters. The page names the affected Table I and II cells (UniVLA WidowX fusion 87.50; UniVLA Spatial fusion 95.00).
5. **Run-to-run noise** on the four WidowX pairs (9 to 17 percent of zero-fire episodes end differently) and the **Benjamini-Hochberg** result (16 of 22 drops pass, no gain passes). Both are the paper draft's analyses, recomputed; the final paper reads the p-values as descriptive and does not print these numbers.
6. **Depth-pruning calibration frame**: for six backbones it is the first observation of the evaluation run, not a disjoint calibration set (CronusVLA has a separate pass with seed 10000).
7. **Foveation**: the paper says tau reaches one at the image boundary; the code reaches one at the farthest corner. The page states the code behaviour. Whether the LIBERO wrist view was foveated is not recorded; the page says so.
8. **Latency rounding**: the page recomputes ms/step from the per-episode records (pooled episode time / pooled steps); Tables I and II of the paper differ by up to 0.1 ms in some cells (e.g. CogACT WidowX original 141.5 on the page, 141.41 in Table I). The page says so in Section 1.3.
9. **LIBERO task names** in Section 2.2 are the standard LIBERO instructions in benchmark order (task 0 to 9); four of them were checked against the rollout videos. Confirm the harness used the benchmark order.
10. **Whether the LIBERO wrist view was foveated** is not recorded and the page does not say; add a sentence if you know.
11. **Software versions**: transformers 4.47.0 for CogACT, CronusVLA, MiniVLA and SpatialVLA (from the attention-backend note); SmolVLA transformers 4.51.3 and lerobot 0.4.4; OpenVLA and UniVLA versions and the torch version are not in the records. If the authors know them, add them to the table in Section 1.6.

## Things that were deliberately left out

- Absolute paths, user names, cluster job ids and the `_mentor` / `_ko` note file names.
- Everything from the earlier three-backbone study (Report_EN.md, Overview_EN.md, `experiments/figures/` except the raw/blur observation pairs), because its numbers do not match the final paper.
- The "pick coke can holds up better than move near" claim from the old notes; the final Fractal records do not support it.
- Any claim that fusion accelerates anything (no cache-level reuse was measured).

## Video badge caveat

Five of the 150 rollout clips used in the videos (three wall tiles and two paired clips, all CogACT WidowX) carry a SUCCESS/FAILURE badge that differs from the episode record, because the WidowX rollout videos are a re-run. The page says under the walls that the clips follow the rendered run.
