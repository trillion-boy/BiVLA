# The five models: what to write now, and what to wait for

*The mentor runs them; results come later. This is what can be written before
they arrive, and how to hold these five papers in the meantime.*

---

## 1. Their role: subjects, not competitors

The efficiency papers (EfficientVLA, VLA-Cache, FastV, VLA-Pruner…) are
**methods we test**. These five are **backbones we test on**. That distinction
decides where they go:

| section | what they get |
|---|---|
| **Setup** | the main home — a backbone table, writable today |
| **Related Work** | three specific paragraphs, all writable today (§2) |
| **Results** | nothing until the runs land |

It also decides the tone. **We make no claim about these five.** We do not
reproduce them, we do not critique their numbers, and their success rates are
not on trial. They are the population over which we ask whether an
intervention's effect holds. Keeping that clean matters: the moment the paper
reads as "we evaluated FLOWER and it did worse," we have picked a fight we did
not intend and cannot win.

---

## 2. Related Work material — better than I expected, and none of it needs results

### (a) The interventions have become architecture

Three of the five build one of our axes into the model:

| model | what it builds in |
|---|---|
| **FLOWER** | *"we prune between **30% and 50%** of the pretrained VLM's layers"* — our depth axis, as a design decision |
| **SmolVLA** | *"We use only the **first 16 layers** of the LLM within the VLM"* — the same axis, as a truncation |
| **TurboVLA** | removes the LLM from the action path entirely: *"avoiding the computation and memory overhead of processing multimodal inputs through a billion-parameter language model"*, chunks emitted *"without autoregressive action-token generation"* |

The paragraph writes itself:

> Recent small VLAs no longer apply these reductions at inference — they bake
> them in. FLOWER prunes 30–50% of its VLM layers by design, SmolVLA keeps
> only the first sixteen, and TurboVLA removes the language model from the
> action path altogether. That makes the question this paper asks more urgent,
> not less: if layer removal is now an architectural commitment rather than a
> switch, whether its reported effect is a property of the method or of the
> configuration it was measured in decides whether the commitment transfers.

### (b) Their own ablations are our thesis — this is the strongest use

We do not have to argue that configuration dominates. **These papers publish
sweeps showing it, about their own models.**

| paper | what they varied | range it produced |
|---|---|---|
| **TurboVLA** Table 3 | language conditioning: none → task-ID → semantic instruction | **70.8 → 95.4 → 97.7** (27 points) |
| **TurboVLA** Table 4 | text encoder: SigLIP-Base → T5-Small → BERT | 95.5 → 97.1 → **97.7** |
| **TurboVLA** | action horizon *H* = 12 vs 15 | drops to 95.6 at H = 15 |
| **SmolVLA** Table 12 | action chunk size 1 / 10 / 30 / 50 / 100 | **50.0 → 84.0 → 78.5 → 80.3 → 74.5** (34 points) |
| **SmolVLA** | action steps executed 1 / 10 / 30 / 50 | 80.3 → 82.8 → 70.8 → **51.8** (31 points) |
| **MiniVLA** | VQ chunk h8 vs h1 (single action) | **77% vs 62.4%** |

Every one of these is the *same shape* as our first result: hold the model
fixed, move one configuration value, watch the number move by tens of points.
The difference is that they report it as a tuning study and move on; we report
it as the finding.

That is a much better Related Work paragraph than "prior work does not report
these values," because it says something stronger: **prior work does report
them, in ablation tables, and then does not carry the implication into how the
main results are read.**

### (c) A fifth author group for the per-task split

MiniVLA's SimplerEnv table is on **our exact four Bridge tasks**, against
OpenVLA:

| | carrot | spoon | stack | eggplant |
|---|---:|---:|---:|---:|
| OpenVLA | 46% | 44% | 62% | 66% |
| MiniVLA + VQ h8 | 44% | **68%** | **70%** | **14%** |
| difference | −2 | **+24** | +8 | **−52** |

A 7× smaller model that matches or beats on three tasks and collapses on the
fourth. Reported by the authors as "3 / 4 tasks," with the −52 not discussed.
That is §2.5's argument in a fifth place, and it is about the models rather
than the methods, so it widens the claim rather than repeating it.

---

## 3. Record the published baselines now — this is the actionable part

When the mentor's results arrive, the first check is **baseline vs. the
model's own paper**. That check catches `unnorm_key` and gripper errors, and it
is only possible if we have the published numbers written down. Here they are.

| model | benchmark | published baseline |
|---|---|---|
| **TurboVLA** (0.2B) | LIBERO Spa / Obj / Goal / Long | 99.2 / 99.8 / 97.4 / 94.2 → **97.7 avg** (0.9 GB VRAM, 31.2 ms) |
| **SmolVLA** (0.45B) | LIBERO | 90 / 96 / 92 / 71 → **87.3 avg** |
| **SmolVLA** (0.24B) | LIBERO | 87 / 93 / 88 / 63 → 82.75 |
| **SmolVLA** (2.25B) | LIBERO | 93 / 94 / 91 / 77 → 88.75 |
| **CoTinyVLA** (0.9B) | **LIBERO-Plus** | 90.8 / 87.3 / 86.6 / 80.7 |
| **FLOWER** (0.95B) | LIBERO, all variants | *"consistently above 93%"*; 50 trials/task (20 for LIBERO-90) |
| **MiniVLA** (~1B) | LIBERO-90 | 62.4% base, **77%** with VQ h8, **82%** with h8 + history or wrist |
| **MiniVLA** | SimplerEnv Bridge | 44 / 68 / 70 / 14 (see §2c) |

### Four traps in that table

1. **SmolVLA ships in three sizes** — 0.24B, 0.45B, 2.25B. None is 4B. The
   headline 87.3 is the 0.45B. Which checkpoint gets run has to be recorded
   next to the result, or the comparison is meaningless.
2. **CoTinyVLA's headline is LIBERO-Plus, not LIBERO.** It is a different
   benchmark — 10,030 perturbed tasks over seven perturbation dimensions. The
   paper says it was *"fine-tuned on the four LIBERO-Plus suites … and
   evaluated on both LIBERO-Plus and the original LIBERO"*, so plain-LIBERO
   numbers exist, but the 90.8 / 87.3 / 86.6 / 80.7 above are **not** them.
   Comparing a plain-LIBERO run against these would manufacture a gap.
3. **FLOWER's LIBERO protocol is 50 trials per task**, not our 5. Its SIMPLER
   Bridge and Google Robot numbers appear only in figures, so they need
   reading off the plot before use — do not quote them from the text.
4. **MiniVLA is a blog post** (`@misc{belkhale2024minivla}`, Stanford, Dec
   2024). Citable and widely used, but it should not be introduced with the
   weight of a reviewed result.

---

## 4. What has to wait for the results

Everything in Results, plus two things that look like Related Work and are not:

- **Whether the effects hold at small scale.** That is the whole point of the
  expansion and cannot be previewed. Write the question now, not an answer.
- **The correction family.** It grows with the new cells, so α is not final
  until we know how many cells actually got filled — and §2 of
  `FiveModels_Read.md` says three conditions do not apply uniformly, so the
  count will be lower than 15 × 7.

---

## 5. How to hold these five papers, in one line each

| model | the sentence to carry |
|---|---|
| **TurboVLA** | 0.2B, no LLM in the action path — so it is the boundary case for whether "VLA" implies a prunable stack at all. Its own ablations move LIBERO 27 points on the language-conditioning choice |
| **SmolVLA** | 450M, three sizes, already truncated to 16 LLM layers. Publishes the chunk-length sweep that makes our action-repeat caveat unavoidable |
| **CoTinyVLA** | 0.9B on Qwen3.5, 16 history frames × 2 views. The one whose headline benchmark is not the one we would run |
| **FLOWER** | 950M, prunes 30–50% of its VLM by design, and the only one of the five with published numbers on **both** our SimplerEnv suites — so it is the cell to run first |
| **MiniVLA** | ~1B, VQ chunking, blog post. Its SimplerEnv table is on our four tasks and contains a −52-point per-task collapse nobody discusses |

**Bottom line.** There is enough here to write the Related Work paragraphs
today, and the strongest material is not the models themselves but their
ablation tables — six published sweeps in which one configuration value moves
a benchmark by 27 to 34 points. That is our thesis, in their own numbers,
before a single new episode is run.
