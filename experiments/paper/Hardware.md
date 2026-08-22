# Which backbone ran on which GPU, and what it lets the paper claim

The result files record no GPU. A search over every file in `results/` for
`gpu`, `device`, `T4`, `L4`, `Tesla` and `NVIDIA` returns nothing, which is why
`Report.md` §7 ⑤ lists the run environment as unrecoverable. What follows is
recovered from two other places instead.

---

## 1. The assignment

Confirmed by the author of the runs, 2026-08-22: **two backbones ran on T4s and
one on an L4**, and the split was forced by memory, since Colab gives about
15 GB on a T4 and 22.5 GB on an L4.

Which one took the L4 is determined by the weights, not by preference:

| backbone | base | params | fp16 weights | fits a 15 GB T4? |
|---|---|---:|---:|---|
| SpatialVLA | PaliGemma 2 | 4B | 8 GB | comfortably |
| OpenVLA | Llama-2 7B | 7B | 14 GB | only just, ~1 GB left for KV cache and context |
| **UniVLA** | Emu3 | **8.5B** | **17 GB** | **no** |

UniVLA cannot fit a T4 in half precision at all, so it is the L4 run. Two
things corroborate this independently of the arithmetic. `Report.md` §3.1
already annotates its latency as **"2.81 s (L4)"**, written when the runs were
fresh. And the parameter counts themselves are from §3.1, which §7.1 records as
having been corrected against the backbone papers.

So: **UniVLA on L4, OpenVLA and SpatialVLA on T4.**

---

## 2. What this changes, and it is not small

The Introduction's strongest cross-backbone result is `depth prune 4` on
Fractal, where **OpenVLA gains $+15.6$ and SpatialVLA loses $-17.8$** on the
same 135 episodes, each surviving correction on its own. Both of those
backbones ran on T4s.

**That comparison is therefore hardware-controlled**, which the paper could not
say before. The same holds for every within-cell contrast, including the
45.9-point candidate-window result, since that is one cell and one backbone.

The comparisons that do cross cards are the ones involving UniVLA, which in our
grid means Bridge only. The action-repeat contrast SpatialVLA $+12.5$ against
UniVLA $-69.8$ is T4 against L4 and should be labelled as such.

It also sharpens the opening sentence. UniVLA is three times slower than
SpatialVLA **while running on the faster of the two cards**, so the gap is a
property of the model rather than of the hardware, and is if anything
understated.

---

## 3. Confirmed, and how much it actually matters

The author confirms the assignment held for every run: the T4 backbones always
used a T4 and UniVLA always used an L4. So a backbone never changed card
mid-campaign, and the claims in §2 stand.

**How much does a card change matter?** Less than the limitation section
implies, and we have a direct measurement rather than a worry. `Report.md` §3.4
re-ran UniVLA/Bridge on a different card:

| condition | mean | per task |
|---|---|---|
| baseline | 78.1% → 78.1% | **0 of 4 tasks moved, not one episode changed** |
| log-polar | 86.5% → 86.5% | **all four moved** (−4.2 / +4.2 / +8.3 / −8.3) and cancelled |
| blur | 76.0% → 72.9% | three moved, `spoon` by −20.8 |

So the policy itself is numerically stable across hardware. What moved was the
**foveation image path**, and §3.4 records that we could not separate a GPU
cause from a `cv2` build difference. Either way the operative rule is the same
and narrower than "fix the GPU": a condition and its baseline must come from
the same environment.

**Interruptions did not damage anything.** Colab runtimes were cut for time on
occasion. Three checks over `results/`:

| check | result |
|---|---|
| episode counts | 24 / 25 / 60 throughout, exactly the protocol |
| duplicate `ep_id` within a task | none |
| a task split across two sessions, visible as a persistent latency level shift between its first and second half | **0 of 139** files with latency records |

So an interrupted task was re-run whole rather than resumed and merged, which
is the right handling and leaves no partial file behind.

**One residual, and it is not about the card.** §3.4.0 notes from the commit
dates that a cell's conditions span several sessions, for instance
SpatialVLA/Bridge with its baseline on 08-05 and conditions from 08-06 to
08-10. Same card type, different session, so possibly a different library
build. Given that the measured instability lives in the `cv2` path rather than
in the policy, this is the residual worth stating in Limitations, not the GPU
model.

### Not a defect: the pruned layers differ per task

Worth recording because it looks alarming in the files. Within one campaign the
deleted layers vary by task, for example SpatialVLA `depth_prune4` removing
[9,10,12,19] on `stack_cube` and [8,9,10,20] on `eggplant`. That is **by
design**: Block Influence is calibrated once per run and a run is one task, so
each task ranks its own layers. `Report.md` §3.5 already records the layers
each run actually deleted for this reason. It is not evidence of recalibration
caused by an interruption, and the Introduction's contrast holds the layer
*count* fixed at four, not the identities.

---

## 4. What `Report.md` should say now that it is confirmed

Not edited yet, because these passages are covered by `verify_all.py` and
should move together with it.

| passage | what it says now | what it would become |
|---|---|---|
| §3.4.0 | bounds the hardware effect at 3.1 points, from two UniVLA baselines differing by 11 of 96 episodes | both of those UniVLA runs were on the **same** L4, so 3.1 points bounds **run-to-run nondeterminism**, not hardware. The bound still holds and its meaning gets cleaner |
| §7 ⑤ | the run environment was never recorded and cannot be recovered | still true of the files. Add that the per-backbone assignment is recoverable from the memory footprint and is recorded here |
| §3.1 | annotates UniVLA "(L4)" without saying where that came from | say it, and add T4 for the other two |

---

## 5. The lesson worth keeping

`Report.md` §7 ⑤ already draws it: **print the GPU, driver and library versions
into every result file.** One line at write time would have made this whole
document unnecessary. The five-model expansion should do it from the first run,
and `notebooks/05` writes a `gpu` field for exactly this reason.

---

## 6. The UniVLA baseline is settled, and a note on how it was described

Recorded because a summary of this file called it "undetermined", which was
wrong and alarming.

There are two UniVLA/Bridge baseline runs and the grid uses the second:

| run | successes | rate |
|---|---:|---:|
| `baseline` | 75 / 96 | 78.1% |
| **`baseline_l4`** | **78 / 96** | **81.2%**, and `Report.md` §3.4.0 names it as the value the grid uses |

They differ on 11 of 96 episodes, which is the 3.1 points that section bounds.
Every condition was tested against **both**, and no conclusion changes. So
**81.2% is our measurement and it is not in question.**

What is a guess is something else entirely: why the UniVLA paper's own table
reports 69.8% where we get 81.2%. §3.8(c)③ traces most of that gap to one task,
`stack_cube` at 75.0% against 33.3%, and suspects a checkpoint difference, but
we could not confirm which checkpoint their table used. That guess is confined
to `Report.md`. Neither 81.2 nor 69.8 appears in `introduction.tex` or
`relatedwork.tex`, which say only that four of five baseline cells sit above
the published figures and one is 4.2 points below.

**The distinction to keep**: our numbers are settled, and what is unconfirmed
is the explanation for a difference with someone else's number.
---

## 7. Every number in the two sections, recomputed from `results/`

Done 2026-08-22 because tracing the `.tex` to `Report.md` only shows the two
agree, and both could be wrong together. This goes back to the episode records.

| claim in the `.tex` | recomputed | |
|---|---|---|
| 7,198 episodes | 7198 | ✓ |
| five baselines 15.6 / 38.5 / 30.2 / 84.4 / 81.2% | identical | ✓ |
| OpenVLA/Fractal prune4 $+15.6$, $p=0.0011$ | $+15.6$, McNemar $p=1.07\times10^{-3}$ | ✓ |
| SpatialVLA/Fractal prune4 $-17.8$, $p=1.8\times10^{-4}$ | $-17.8$, $p=1.82\times10^{-4}$ | ✓ |
| window875 $-30.4$ | $-30.4$, $p=5.89\times10^{-11}$ | ✓ |
| the 45.9-point contrast | $15.6-(-30.4)=45.9$ | ✓ |
| span 2.1 to 50.4 over five cells | 2.1 / 5.2 / 6.2 / 45.9 / 50.4 | ✓ |
| keep sweep $+30.2$ / $+18.8$ / $+4.2$ | identical, and 40% fills in at $+19.8$ | ✓ |
| the two cells differ at $p=4.3\times10^{-7}$ | Fisher 2×2 on (30,9) vs (8,32) gives $4.30\times10^{-7}$ | ✓ |
| compute saved $-11.9\%$ | $-11.9\%$ from `discover_cost` | ✓ |
| SpatialVLA 0.90 s | 902.1 ms | ✓ |
| four baselines above, one 4.2 below | +14.6 / +3.7 / +15.5 / +4.3 / +13.7 / +11.4 and −4.2 | ✓ |
| **UniVLA 2.80 s** | **2811.5 ms = 2.81 s** | **✗ fixed** |

**The one error.** The Introduction took UniVLA's latency from `baseline`
(2801.5 ms) while taking its success rate from `baseline_l4` (81.2%), which is
the run the grid actually pairs against and which averages 2811.5 ms. Two
numbers in the same sentence came from two different runs. `Report.md` §7.1 had
already standardised on 2.81 s; the `.tex` was carrying the older value.
Corrected in `introduction.tex` and `Introduction.md`, and the footnote now
says the figures come from the runs the grid pairs against.

**Two things that looked wrong and were not.** The OpenVLA/Bridge foveation
cell is not under `results/` at all, but in
`RetinaBased/GoogleColab/results_reproduction_eager/`. `build_grid_report.py`
imports it deliberately and checks at load time that the borrowed campaign's
own baseline is episode-for-episode identical to the grid's, which it is at
15/96. And `keep_percent: 20.0` appears in every run's config because it is the
flag's default, not because every run foveates.

---

## 8. What the Introduction handed to Setup, and why

Measured across the 21 papers in this session's reading set, taking each one's
Introduction:

| | papers doing it |
|---|---|
| reporting a p-value in the Introduction | **0 of 21** |
| using a footnote in the Introduction | **0 of 21** |
| naming a GPU in the Introduction | several, but as scale or to make a latency figure legible |

The last row is the useful distinction. MoLe-VLA writes *"a 7B VLA model
running on a commercial-grade GPU like the RTX 4090 generally achieves an
inference frequency of approximately 5−12 Hz"*, and Bag of Tricks writes *"over
20,000 A100-80G GPU hours"*. Naming a card so a number can be read is normal.
Explaining the assignment is not.

**Moved out of the Introduction.**

| what | now belongs in |
|---|---|
| $p = 0.0011$, $p = 1.8\times10^{-4}$, $p = 4.3\times10^{-7}$, $p < 10^{-4}$ | Results |
| that the latency figures come from the runs the grid pairs against | Setup |
| that the result files record no card, and that this is why the grid carries a bound on run-to-run variation rather than a controlled hardware comparison | Limitations |

**Kept, and why.** The three claims those p-values supported all survive
without them: one benchmark-axis comparison passes correction, six
backbone-axis comparisons do, and the two Fractal cells each pass on their own
while differing from each other. An Introduction states what was found. The
tests belong with the results.

The footnote keeps model-time-only, the episode count, which card each backbone
needed, and the published figure that agrees on the scale. That is what makes
2.81 s and 0.90 s readable, and it is the same job MoLe-VLA's sentence does.

**Setup must now carry** the memory table in §1, the per-backbone assignment,
and the confirmation in §3 that no backbone changed card mid-campaign, since
the hardware-controlled claim in §2 rests on it and the Introduction no longer
says it.

---

## 9. The compute figures, and whether the window contrast really holds compute fixed

Measured 2026-08-22 because the Introduction says the $-10.6$ to $-11.9\%$
spread is *"no wider than repeating the same measurement produces"* and
nothing in `verify_all.py` checked that. The determinism checks there are on
**success outcomes** (85/85 and 24/24 episodes identical), not on time.

**What compute is.** `build_grid_report.discover_cost` reads
`model_ms_per_infer`, so the compute column is **milliseconds, not FLOPs**.
That is why a spread can be noise at all. A FLOP count would be deterministic
and a $1.3$ point gap would have to mean something.

**The two window settings, OpenVLA/Fractal, 135 episodes each.**

| condition | mean ms per call | sd | against baseline |
|---|---:|---:|---:|
| baseline | 515.5 | 3.9 | |
| `depth_prune4` | 454.3 | 7.5 | $-11.9\%$ |
| `window875` | 460.8 | 4.1 | $-10.6\%$ |

The two settings differ by $6.5$~ms, which is $1.3\%$ of the baseline.

**What repeating the same measurement produces.**

| pair | mean ms | mean ms | difference |
|---|---:|---:|---:|
| SpatialVLA/Fractal `baseline` vs `baseline_rerun` | 936.8 | 914.4 | $-2.4\%$ |
| UniVLA/Bridge `baseline` vs `baseline_l4` | 2801.5 | 2811.5 | $+0.4\%$ |

**So the sentence holds.** Re-running the same condition moved the mean by
$2.4\%$ in one case, which is larger than the $1.3\%$ separating the two
window settings. The claim that compute is held fixed across the $45.9$-point
contrast survives.

**Two caveats worth keeping.** The $2.4\%$ comes from a different backbone and
benchmark, since OpenVLA/Fractal has no re-run of its own, so it is a proxy
rather than a direct bound. And the within-condition sd is only $4$ to $8$~ms,
so the variation lives between sessions rather than inside one. The honest
form of the claim is that the gap is smaller than what a session change moves,
which is what the Introduction says.
