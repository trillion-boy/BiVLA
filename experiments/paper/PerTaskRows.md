# The fifteen rows behind "fourteen of the fifteen configurations"

Written 2026-08-22 because the footnote said *"twelve rows
from~\cite{efficientvla}"* and gave a reader no way to find them. If the
author of the paper cannot locate the rows from the footnote, neither can a
reviewer. Every row is listed here with its arithmetic.

The claim in the Introduction is that `pick coke can` holds up better than
`move near` under efficiency interventions, and that this split is already
visible in other people's published tables rather than being an artefact of
our setup.

---

## 1. Where the twelve rows are

**EfficientVLA, Table 2**, titled *"Performance of EfficientVLA on the CogACT
versus the other baselines in the SIMPLER environment."*

The table has two blocks, `Visual Matching` and `Variant Aggregation`. Each
block opens with `CogACT`, which is the **unmodified backbone** and not an
intervention, then lists seven rows. One of those seven is `Random Dropping`,
which the caption defines as a control. So each block contributes **six**
intervention rows, and 6 + 6 = 12.

The comparison is each row's `PickCan` and `MoveNear` against the `CogACT` at
the head of **its own block**, since the two blocks have different baselines.

### Visual Matching, against CogACT at PickCan $91.3$, MoveNear $85.0$

| row | PickCan | MoveNear | Δ Pick | Δ Move | holds |
|---|---:|---:|---:|---:|:-:|
| FastV | 92.6 | 81.4 | +1.3 | −3.6 | ✓ |
| VLA-Cache | 92.0 | 83.3 | +0.7 | −1.7 | ✓ |
| EfficientVLA (L=28, T=112) | 95.3 | 83.3 | +4.0 | −1.7 | ✓ |
| EfficientVLA (L=28, T=56) | 94.7 | 82.4 | +3.4 | −2.6 | ✓ |
| EfficientVLA (L=22, T=112) | 94.0 | 82.1 | +2.7 | −2.9 | ✓ |
| EfficientVLA (L=22, T=56) | 93.3 | 81.3 | +2.0 | −3.7 | ✓ |

### Variant Aggregation, against CogACT at PickCan $89.6$, MoveNear $80.8$

| row | PickCan | MoveNear | Δ Pick | Δ Move | holds |
|---|---:|---:|---:|---:|:-:|
| FastV | 91.4 | 78.6 | +1.8 | −2.2 | ✓ |
| VLA-Cache | 91.7 | 79.3 | +2.1 | −1.5 | ✓ |
| EfficientVLA (L=28, T=112) | 94.8 | 77.6 | +5.2 | −3.2 | ✓ |
| EfficientVLA (L=28, T=56) | 94.4 | 77.2 | +4.8 | −3.6 | ✓ |
| EfficientVLA (L=22, T=112) | 93.9 | 76.4 | +4.3 | −4.4 | ✓ |
| EfficientVLA (L=22, T=56) | 93.2 | 75.8 | +3.6 | −5.0 | ✓ |

**Twelve of twelve hold.** `PickCan` rises in every one of them while
`MoveNear` falls in every one of them, which is a stronger statement than the
Introduction makes.

---

## 2. The two excluded rows, and why the exclusion is disclosed

| row | PickCan | MoveNear | Δ Pick | Δ Move | holds |
|---|---:|---:|---:|---:|:-:|
| Random Dropping, Visual Matching | 9.7 | 20.4 | −81.6 | −64.6 | ✗ |
| Random Dropping, Variant Aggregation | 4.0 | 16.1 | −85.6 | −64.7 | ✗ |

Both break the pattern, so excluding them is exactly the move a reviewer
should be suspicious of. Two things make it defensible, and the footnote says
both.

**It is the paper's own classification, not ours.** The Table 2 caption reads
*"Random Dropping denotes a method involving the random retention of 112
visual tokens,"* and the rows are set in grey italic where every proposed
method is set in roman. It is the destroy-the-signal control that shows the
other rows are doing something.

**The direction is what the claim is about.** At −81.6 the policy is not
performing the task at all, so "which task degrades more gracefully" has no
content. A row where both tasks have collapsed cannot inform a claim about
relative graceful degradation.

Either way it is in the footnote, so a reader who disagrees can see the rows
and discount the claim themselves.

---

## 3. The other three rows, and the one exception

Three rows come from **VLA-Pruner**, giving fifteen rows and four method
families. Fourteen of the fifteen hold, so the single exception is in these
three.

The Introduction attributes it to the one method built to preserve
action-relevant tokens, which is what VLA-Pruner's own dual-level mechanism
predicts, and `TableI_Cells.md` records that mechanism from the paper.

⚠️ **These three rows are not re-derived here.** They were checked against the
PDF when the sentence was written, but the flattened text used for that check
lived in a session scratchpad and is gone, and the PDF is not in the
repository. The twelve rows in §1 are re-derived above from the table itself.
**Before submission, re-open VLA-Pruner's Table 2 and confirm the three rows
and the identity of the exception.** It is the one number in this footnote
that rests on a note rather than on arithmetic anyone can repeat.

---

## 4. Why the claim is worth its footnote at all

Our own finding is that the backbone axis moves results more than the
benchmark axis, measured on 135 paired Fractal episodes. That stands on its
own data.

What these fifteen rows add is that **the same task-level split is already
sitting in other people's published tables**, across four method families and
three independent author groups, and that not one of those papers remarks on
it. That is the difference between "our setup produced an odd result" and "the
field has been averaging over this for years." It is worth four lines.

The honesty cost is the footnote, which has to say that two rows were removed
and that the FastV rows are reproductions rather than FastV's own results.
Both disclosures are in it.
