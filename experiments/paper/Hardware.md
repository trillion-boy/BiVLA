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

## 3. One thing still to confirm

The author's statement fixes a card per *backbone*. What the paper needs is
slightly stronger: that a backbone used **the same card for every one of its
runs**, baseline and conditions alike. If a backbone was ever moved between
cards mid-campaign, a within-cell contrast could straddle two GPUs and the
claim in §2 weakens.

Nothing in the files can settle this, since the files hold no GPU field. It
rests on how the campaigns were actually launched.

Until it is confirmed, the safe wording is what the Introduction footnote now
uses: name which card each backbone's weights required, and keep the grid's
existing bound on run-to-run variation rather than claiming a controlled
hardware comparison.

---

## 4. What `Report.md` should say once it is confirmed

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
