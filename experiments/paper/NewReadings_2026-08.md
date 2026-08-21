# Three more papers read from source — and what changes

VLA-Pruner, SparseVLM, DeeR-VLA, read from the PDFs. Two of the three settle a
training-free question. The third does more than that: **VLA-Pruner changes two
claims we were going to make.**

---

## 1. Training-free verdicts

| paper | the sentence that settles it | verdict |
|---|---|---|
| **SparseVLM** (ICML'25) | Abstract: *"most existing methods **learn a network** to prune redundant visual tokens using certain training data. **Differently**, we propose a text-guided **training-free** token optimization mechanism … that **eliminates the need of extra parameters or fine-tuning costs**."* | ✅ training-free |
| **VLA-Pruner** | §1: *"We propose VLA-Pruner, a general, **training-free** framework."* §4: *"VLA-Pruner is **training-free**, serving as a **plug-and-play** acceleration module."* Its only two uses of "loss" are *accuracy loss* and *without loss of generality* — neither is an objective | ✅ training-free |
| **DeeR-VLA** (NeurIPS'24) | Abstract: *"we design a tailored **training method** … on top of such multi-exit architectures."* §: N auxiliary action heads at the exits, *"We **jointly train** the auxiliary heads and the MLLM"* with $\mathcal{L}_{\text{aux}}$ (Eq. 8). *"We **fine-tune** … the perceiver sampler and cross-attention layers … with the randomly initialized action head."* Appendix A.3: a two-phase training schedule | ❌ **not training-free** |

My earlier "learned exit criterion, provisional" for DeeR-VLA was right and is
now verified. Independent confirmation, from VLA-Pruner's own related work:
early-exit methods *"often require architectural modifications or retraining."*

So the disqualified list is now **MoLe-VLA and DeeR-VLA**, both confirmed from
source, and nothing on the candidate list is unverified any more.

---

## 2. VLA-Pruner is a counterexample to our twelve-configuration claim

We say `pick coke can` rises or holds while `move near` falls in **all twelve**
configurations published across EfficientVLA, VLA-Cache and FastV. VLA-Pruner's
Table 2 adds three more configurations on SIMPLER (OpenVLA, 75% pruning), and
one of them goes the other way.

Relative accuracy preserved against the unpruned baseline:

| method | Move Near | Pick Coke Can | which held up better |
|---|---:|---:|---|
| FastV | 71.7% | **79.4%** | Pick Coke ✅ matches our pattern |
| VLA-Cache | 76.3% | **76.9%** | Pick Coke ✅ (but 0.6 pt — effectively tied) |
| **VLA-Pruner** | **97.0%** | 94.9% | **Move Near ❌ breaks the pattern** |

*(Absolute rates: baseline 54.0 / 52.8; FastV 38.7 / 41.9; VLA-Cache 41.2 /
40.6; VLA-Pruner 52.4 / 50.1. Ratios recomputed and they match.)*

**This is not a refutation — it is a better version of the claim.** VLA-Pruner
is the one method in the table *designed* to stop pruning action-critical
tokens; its whole premise is that semantic-salience pruning "may remove
action-critical visual tokens." So the pattern holding for the
salience-based methods and breaking for the method built to fix it is exactly
what the mechanism predicts.

The claim should become:

> Across fifteen published configurations from four author groups, `pick coke
> can` holds up better than `move near` in fourteen. The single exception is
> the one method explicitly designed to preserve action-relevant tokens.

That is stronger than 12/12, because it now has a mechanism attached rather
than just a count. **The Introduction and Related Work both currently say
"twelve" and must be updated.**

---

## 3. VLA-Pruner also changes the competitor table

`RelatedWork.md` §2.6 claims no prior paper crosses a backbone axis and a
benchmark axis. VLA-Pruner has **both**:

| | LIBERO (4 suites, 500 ep each) | SIMPLER (3 tasks) | real xArm6 |
|---|:--:|:--:|:--:|
| OpenVLA | ✓ | ✓ | — |
| OpenVLA-OFT | ✓ | **✗** | ✓ |
| π0 | ✓ (appendix) | ✗ | ✗ |

Same shape as Gaze-Reg: both axes present, **the crossing cell empty**. Our
claim survives — nobody measures the same intervention across both axes — but
the §2.6 table needs a new row and the hedge needs to be firmer, because this
is a paper we had not read and it is closer than the ones we had.

Two more things worth taking from it:

- **It reports LIBERO per suite and SIMPLER per task**, so "everyone reports
  only the four-task average" is too broad. The correct statement is narrower
  and still true: *the Google Robot four-task average is what EfficientVLA,
  VLA-Cache and FastV report, and the split inside it is not discussed.*
- **Its own hyperparameters are "empirically set"** — window $w = 3$, decay
  $\gamma = 0.8$, plus a warm-start of $w$ steps — with no sweep shown. That is
  another instance of the pattern our first result is about, and it belongs in
  Table I.

---

## 4. More published evidence on FastV, in our favour

VLA-Pruner independently reproduces the FastV-on-VLA degradation that
VLA-Cache reported, on a third setting:

| source | what FastV does |
|---|---|
| FastV's own paper | −45% FLOPs on LLaVA-1.5-13B, "without sacrificing performance" |
| VLA-Cache | on OpenVLA: FLOPs unchanged, latency **up** (51.91 → 53.28 ms) |
| **VLA-Pruner** | on OpenVLA/SIMPLER at 75%: **73.1%** of baseline accuracy retained; on LIBERO it is the weakest baseline in Table 1 |

So the same method is reported as near-free on a VLM and as costly on VLAs, by
two independent groups. That is the cleanest published instance of our thesis
that exists, and it is about a method we already have implemented.

---

## 5. What has to change in the drafts

| file | change |
|---|---|
| `introduction.tex`, `Introduction.md` | "all twelve configurations … two author groups and three method families" → fifteen configurations, four groups, fourteen consistent, with the exception named |
| `RelatedWork.md` §2.5 | same count fix, plus the mechanism sentence |
| `RelatedWork.md` §2.6 | add VLA-Pruner as the closest competitor; keep the "we are not aware of" hedge and strengthen it |
| `MethodAxes_Survey.md` | SparseVLM and VLA-Pruner verified training-free; DeeR-VLA disqualified from source |
| Table I | add VLA-Pruner as a row; its $w = 3$, $\gamma = 0.8$ are "empirically set" with no sweep |
