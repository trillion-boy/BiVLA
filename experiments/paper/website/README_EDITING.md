# Project page: how to edit and rebuild

Everything on the page is generated from the files in this folder. `site/` is the output; do not edit it by hand unless you only want a quick one-off change (then edit `site/index.html` directly in any text editor).

## Rebuild

    pip install markdown        # once
    python3 build.py            # writes site/index.html and copies assets

Then open `site/index.html` in a browser. Upload the contents of `site/` (index.html, assets/, figs/) to the root of the anonymised repository.

## Where each part of the page lives

| Page part | File |
|---|---|
| Intro paragraph under the title | `content/intro.md` |
| 1. Additional implementation details (1.1 to 1.3) | `content/impl.md` |
| 2. intro sentence | `content/results_intro.md` |
| 2.1 note above the full tables | `content/full_table_note.md` |
| 2.1 tables themselves | generated from `data/full_settings.csv` |
| 2.2 note and tables | `content/per_task_note.md`, `data/per_task.csv`; LIBERO task names are in `build.py` (`LIBERO_TASKS`) |
| 2.3 intro and figure captions | `content/figures_intro.md`, `figures.json` (captions, alt text) |
| 2.4 statistics | `content/stats.md` |
| 2.5 rollout clips and captions | `walls.json`; the clips are in `walls/` |
| 3. Additional discussions | `content/discussion.md` |
| Videos section, headings, footer, section order | `build.py` (the `page = f'''...'''` block near the end) |
| Colours, fonts, spacing | `style.css` |

The Markdown is plain: `**bold**`, pipe tables, one blank line between paragraphs. HTML entities and tags are allowed (the foveation formula uses `√`, `ρ`, `<i>`).

## Data

`data/full_settings.csv` (308 rows) and `data/per_task.csv` (2,296 rows) were computed from the per-episode records (`results_corrected/**/episodes.jsonl` and `summary.json`) by `website_data/build_results_data.py` in the repository. Column `in_table` marks the setting printed in Tables I and II (from `paired_results_all.csv`). Both CSVs contain no names or paths and are also offered for download on the page.

## Figures and videos

`figs/` holds the four figures used (consistency, success change Fractal and LIBERO with backbone titles, foveation examples). `media/` holds the two page videos (3-minute and extended). `walls/` holds the ten rollout-wall clips and their poster frames.

## Notes

`NOTES_FOR_AUTHORS.md` lists the statements that go beyond the paper and that the authors should confirm before publishing. Nothing in this folder except the contents of `site/` should be uploaded to the anonymised repository.
