# Table I cells — what prior work actually reports

*Extracted from the five PDFs, page-by-page, on 2026-08-22. Every cell below
carries a page number and a quote. The negative cells were established by
grep over the full extracted text, not by failing to notice something.*

> **Read the headline first: the premise this table was planned on is wrong.**
> `PaperPlan.md` §2 predicted *"mostly empty cells, and each empty cell is one
> of the values we show changes the answer."* That is **not** what the PDFs
> say. Four of the seven columns are essentially full, and three of the five
> papers ablate the very knob we claimed goes unreported. Only the last two
> columns are empty — and those are empty **uniformly, across all five**.
> §4 below reframes the table so that it says something true.

---

## 1. The table, as the PDFs actually support it

| | candidate scope | selection constraint | keep value | knob ablated? | per-task split | per-episode records | paired test |
|---|---|---|---|---|---|---|---|
| **ShortGPT** | **all layers** | **none** (top-*n* by BI) | 25% of 40 | ✓ ratio sweep (Fig. 6) | per-benchmark | ✗ | ✗ |
| **EfficientVLA** | **all layers** ℓ∈{1,…,N} | **none** — "non-contiguous" | L=28/22, T=112/56 | ✓ L×T grid | ✓ 4 tasks | ✗ | ✗ |
| **FastV** | **K**, the start layer | n/a (token axis) | R = 50/75/90% | ✓ **Fig. 7, K×R** | ✓ Table 1 | ✗ | ✗ |
| **VLA-Cache** | layer-adaptive α_ℓ | τ=0.996, k=100, τ_task=0.5 | k = 100 | ✓ k and τ tables | ✓ Table 6 | ✗ | ✗ |
| **VLA-Pruner** | token budget k̃ | dual-level (semantic+action) | 50 / 75 / 87.5% | ✓ ratio sweep | ✓ | ✗ | ✗ |

---

## 2. Cell-by-cell provenance

### ShortGPT

- **candidate scope = all layers; constraint = none.** §3.2, p4:
  > *"we calculate the BI score based on the collected hidden states. Finally,
  > we sort layers in ascending order according to the BI, and delete the
  > layers with the lower BI score."*

  No window, no spacing rule. Grep over the full text: `window` 0,
  `gap` 0, `consecutive` 0, `contiguous` 0, `adjacent` 0.

- **It goes further than any other paper here — Table 9 (p15) lists the
  concrete removed layer indices.**
  > Llama-2-7B: `27, 26, 25, 28, 24, 29, 23, 21, 22`
  > Llama-2-13B: `33, 31, 32, 30, 29, 34, 28, 35, 27, 26`

  **This is the most useful single fact we recovered.** Llama-2-7B has 32
  layers (0–31). BI's own ranking selects **21–29** and never touches 30 or
  31. So "all layers are candidates" *in specification* behaves as
  "deep-but-not-final" *in practice*, and the paper never remarks on it. The
  same holds for the 13B (40 layers, removes 26–35, never 36–39).

- **Table 10 (p16)** names the four strategies compared — Sequential,
  Reverse-order, Relative Magnitude, BI — so the alternatives are on record.

### EfficientVLA

- **candidate scope = all layers; constraint = explicitly none.** §3.2.2
  *"Importance-Driven Non-Contiguous Layer Pruning"*, p5:
  > *"Based on these importance scores, we employ a non-contiguous pruning
  > strategy. For an LLM comprising N layers, the importance score I(ℓ) is
  > computed for every layer ℓ∈{1,…,N}. These scores are then sorted in
  > ascending order … Subsequently, the first n layers from this list … are
  > selected for removal."*

  **"Non-contiguous" means the result need not form a contiguous block — it
  is the absence of a constraint, not a spacing rule.** Worth stating
  plainly, because the section title reads like the opposite.

- **keep values, p7**: *"Settings vary by retained LLM layers (L) and visual
  tokens (T)"* — L=28/22 × T=112/56 as a grid, plus *"Random Dropping denotes
  a method involving the random retention of 112 visual tokens."*

- **per-task split: present.** The SIMPLER table (p7) has columns
  `PickCan | MoveNear | Drawer | DrawerApple | Average`, e.g.
  EfficientVLA (L=28, T=112) → 95.3 / 83.3 / 70.3 / 56.5 / **76.4**.

  ⚠️ **This corrects our own note.** `RelatedWork.md` A.1 says *"보고 단위가
  4-태스크 평균이다"*. The headline unit is the average, but the per-task
  numbers **are** printed. Anything we write must not say they only report
  the mean.

### FastV

- **The start layer is a first-class, ablated parameter.** §4.1, p7:
  > *"It consists of one ranking function f_φ and two parameters: filtering
  > layer K and filtering ratio R%. At layer K of the LVLM, the ranking
  > function f takes a sequence of input tokens and rank them … The last R%
  > tokens after ranking would be pruned out in successive layers."*

- **Fig. 7 (p11) is an explicit K×R ablation**, and its caption states the
  interaction:
  > *"When K is small, lowering R would improve the performance with a
  > smaller FLOPs reduction ratio. In contrast, when K is large, changing R
  > has minimal impact on the overall performance."*

  **FastV already demonstrates, on the VLM side, that where you start
  decides how much the keep-ratio matters.** We cannot claim the field is
  unaware of this. We can claim nobody has measured it on the VLA side.

- **per-task split: present.** Table 1 (p9) reports Nocaps / Flickr30k /
  A-OKVQA / MMMU separately, with K and R as columns.

### VLA-Cache

- **All three hyperparameters are given, p14:**
  > *"Unless specified otherwise, we use a static token similarity threshold
  > τ = 0.996, top-k = 100 for retained static tokens, and a task-relevance
  > threshold τ_task = 0.5. These parameters are applied consistently across
  > all simulated [settings]."*

- **Both are swept**, with full tables: k ∈ {50, 80, 100, 120} at τ=0.5, and
  τ ∈ {0.2, 0.3, 0.4, 0.5} at k=100, on LIBERO-Spatial with OpenVLA-OFT.
  Default declared: *"Our default (k=100, τ=0.5) is used for all main results."*

- **per-task split: present**, p14: *"Table 6 presents detailed results on
  each subtask in the LIBERO-Spatial suite."*

- **Hardware is reported** (RTX 4090, BF16, DDIM 10 steps, CFG 1.5) — worth
  noting given our own GPU-model limitation.

### VLA-Pruner

- **Budgets explicit**, p6: *"All methods are evaluated under identical given
  pruning budgets. To ensure a comprehensive comparison, our experiments
  include diverse pruning ratios (50%, 75%, 87.5%). For VLA-Cache, we set the
  same token reuse ratio."*
- **Episode counts explicit**, p6: *"Each suite contains 10 distinct tasks
  with 50 evaluation episodes per task (500 total episodes per suite)."*
  Real robot: 100 trials per task.
- Of the five, this is the **most careful on protocol** — it equalises the
  budget across methods before comparing, which is the control our own
  §4.4(c) contrast relies on.

---

## 3. The two columns that really are empty

Verified by grep over the **complete** extracted text of all five papers:

| term | ShortGPT | EfficientVLA | FastV | VLA-Cache | VLA-Pruner |
|---|---|---|---|---|---|
| `McNemar` | 0 | 0 | 0 | 0 | 0 |
| `paired` | 0 | 0 | 0 | 0 | 0 |
| `error bar` | 0 | 0 | 0 | 0 | 0 |
| `confidence interval` | 0 | 0 | 0 | 0 | 0 |
| `standard deviation` / `std` | 0 | 0 | 0 | 0 | 0 |
| `seeds` | 0 | 0 | 0 | 0 | 0 |
| `per-episode` / `episode-level` | 0 | 0 | 0 | 0 | 0 |
| `raw results` / `per-trial` | 0 | 0 | 0 | 0 | 0 |

**No paired testing, no dispersion of any kind, no episode-level records, in
any of the five.** This is the one claim the table can carry at full strength,
and it is the one our method contribution rests on.

**Re-verified 2026-08-22 with a wider term set**, because the first grep never
covered plain statistics vocabulary. `significan*`, `p-value`, `p < 0.`,
`t-test`, `chi-square`, `wilcoxon`, `bootstrap`, `confidence interval`,
`error bar`, `standard deviation`, `std`, `seed`, `statistically` over the
complete text of all five PDFs. Every hit is either the colloquial
"significantly reduces" or citation noise, namely "Bootstrapping" inside BLIP
titles and "Seed-Bench". Zero statistical tests of any kind, in any of the
five, under both term sets.

---

## 4. What Table I should become

The original framing — *"they don't report the knob"* — is dead. Three
replacements, in order of how well the evidence supports them.

**(a) The strongest, and fully supported.** *Every one of these papers reports
and ablates its knob, and not one reports an episode-level outcome or a
dispersion.* So the field is careful about **configuration** and silent about
**uncertainty**. Our contribution is not "we report the knob they hid" but
"we test whether a difference is real at all." Columns 4 and 6–7 of §1 carry
this on their own.

**(b) The specification leaves a choice open, and the choice decides the
sign. Check resolved — see the wording constraint.**

Both layer papers specify *all layers, rank by BI, cut the lowest n, no
spacing rule.* Neither says anything about restricting the candidate set,
because in their formulation there is nothing to restrict.

**The check flagged here is now done, and it went against the stronger
wording.** `--depth-min-layer` is **our** flag: defined in
`adaptive_sparse_vla/eval_libero.py:273` and
`RetinaBased/PythonProject/simple_eval.py:244`, both first committed to this
repository (2026-07-12 and 2026-07-27). It is not inherited from the OpenVLA,
SpatialVLA or UniVLA upstreams. Its own help text carries the reasoning we
applied at the time:

> *"only layers past this fraction of the stack are eligible; early layers
> carry too much to bypass"*

So the honest sentence is **"an implementation choice the specification leaves
open"** — *not* "a flag implementations disagree on." We have no evidence
about what other people's implementations do, and must not imply we do.

What survives, and it is enough: a restriction that looks obviously sensible,
that neither method specifies, and that a careful implementer would add
without recording it, moves the same intervention from **+15.6 to −30.4** with
all five other variables held fixed (`Report.md` §4.4(c)). Within our own
three harnesses the same flag name is already read two ways — a **ratio** on
OpenVLA/UniVLA, a **count** on SpatialVLA, which additionally protects the
last layer — and `Report_EN.md` §6 records that this collision produced a
false counterexample we had to retract. That is first-hand evidence that the
choice is easy to get wrong, which is the claim we can actually defend.

**(d) The best framing, and it only became available with Gromov. The
constraint column is not empty — it is *contested*, and nobody has settled it
on a VLA.**

Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers*,
**ICLR 2025**, takes the opposite position to both VLA layer papers on both
knobs:

| | ShortGPT / EfficientVLA | Gromov et al. |
|---|---|---|
| candidate set | all layers | **deepest layers only** |
| contiguity | none — top-*n* by BI, explicitly *"non-contiguous"* | **contiguous block**, and non-contiguity argued against |
| final layer | no special treatment | **excluded by rule** |

Quotes, with pages:

- **Contiguous, and non-contiguity rejected** (p17):
  > *"Liu et al. (2023a) considered non-contiguous pruning proposals, e.g.
  > dropping alternate layers. Our intuition for layer pruning predicts that
  > this shouldn't work as well … as it creates multiple mismatches, one with
  > each block of layers removed."*

- **The final layer is excluded by construction** (p5):
  > *"drop the deepest layers, excluding the final layer before the LLM head …
  > if we are pruning n layers from an L-layer model, then we would remove
  > layers (L−n) to (L−1), inclusive."*

- **And it is stated as a finding, not a convention** (p16):
  > *"for all pruning sizes keeping the very last layer is essential."*

- **With the mechanism** (p8, Fig. 4):
  > *"the deeper layers tend to be very similar, though the deepest blocks
  > that include the final layer … are (near-)maximally dissimilar."*

**This explains our worst cell.** `window875` restricts OpenVLA/UniVLA to
L28–31 — exactly four candidates for four deletions, so BI does nothing and
the final layer L31 is removed by force — and it scores **−30.4**
(`Report_EN.md` §4.4(b)). Gromov predicts that outcome and gives the reason.
Our −30.4 is not an anomaly to explain away; it is an independent
confirmation, on a VLA, of an ICLR 2025 result about LLMs. And SpatialVLA,
the one backbone whose implementation protects the last layer, is the one
that could not run the condition at all.

So Table I's second column becomes the most interesting one in the table: two
groups specify **opposite** constraints, each on its own evidence, neither on
a robot policy. That is a gap no amount of re-reporting closes — it needs the
measurement, and we have it.

⚠️ **Gromov is not training-free.** `training-free` 0, `without training` 0;
the method heals with QLoRA finetuning, *"non-optionally"* for the simple
variant. It publishes no-healing curves throughout (Figs. 2–3), which is what
makes it comparable to us, but it must **never** appear in a training-free
cite list. Our current use — `\cite{shortgpt, gromov}` for *"layer redundancy
is well established"* — is a redundancy claim, not a training-free one, and is
safe as written.

**(c) ShortGPT's Table 9 — the nicest single piece of evidence.** The
specification says all layers are candidates; the method's own published
selection is 21–29 of 32, never the final two. So even the paper that reports
the most has an *implicit* window that it never names, and our result says
where that window sits is worth 46 points. This is one sentence plus one
citation, and it needs no additional run.

---

## 5. Consequences for other documents

| document | what needs changing |
|---|---|
| `PaperPlan.md` §2 | *"Mostly empty cells"* is false. Rewrite the Table I description around §4(a). |
| `RelatedWork.md` A.1 | *"보고 단위가 4-태스크 평균"* — add that per-task numbers are printed too. |
| `relatedwork.tex` | The matched-episode sentence is **confirmed** by §3 above and needs no change. |
| `ICRA_Readiness.md` | Any item asserting prior work hides its configuration should be re-read against this file. |

---

## 6. Housekeeping

- **The two `VLAPruner.pdf` uploads are the same work under two titles** —
  *"VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning…"* (17 p) and
  *"Bridging the Semantic-Action Gap in Visual Token Pruning…"* (20 p). Both
  call the method "VLA-Pruner" ~95 times and share the 73.1% figure. Not a
  citation error; the `.bib` just has to match whichever version we cite.
- The two `FastV.pdf` uploads are byte-identical.
- Text extracted with PyMuPDF; the flattened copies used for grep live in the
  session scratchpad and are **not** committed — the PDFs are not ours to
  redistribute.
