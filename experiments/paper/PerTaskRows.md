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

**VLA-Pruner, Table 2**, at a 75\% token pruning ratio, against `OpenVLA` at
the head of the table. This table prints a preservation ratio next to each
rate, so the comparison can be read straight off it.

| row | Move Near | Pick Coke Can | margin | holds |
|---|---:|---:|---:|:-:|
| FastV | 71.7% | 79.4% | +7.7 | ✓ |
| VLA-Cache | 76.3% | 76.9% | +0.6 | ✓ |
| VLA-Pruner | 97.0% | 94.9% | −2.1 | ✗ |

Raw points give the same verdict. Against `OpenVLA` at Move Near $54.0$ and
Pick Coke Can $52.8$, FastV drops $-15.3$ against $-10.9$, VLA-Cache $-12.8$
against $-12.2$, and VLA-Pruner $-1.6$ against $-1.7$.

**The exception is VLA-Pruner itself**, which is what the Introduction says
and what its own dual-level mechanism predicts, since it is built to keep
action-relevant tokens.

**12 + 2 = 14 of 15. The sentence is confirmed.**

### The soft spot, which the sentence does not hide but does not surface either

The margins are not uniform, and the two smallest are in this table.

| source | rows | margin range |
|---|---:|---|
| EfficientVLA Table 2 | 12 | $+2.4$ to $+8.7$ |
| VLA-Pruner Table 2 | 3 | $+7.7$, $+0.6$, $-2.1$ |

So thirteen rows carry a clear margin, and two are close enough to a tie that
a reviewer could call them one. Both of those sit in the same table, and there
is a coherent reason. At 75\% retention VLA-Pruner preserves $96.8\%$ of
overall performance, so those rows barely degrade at all, and a row that does
not degrade cannot show how degradation splits across tasks.

That reason is real, but turning it into an exclusion rule would add a second
filter to a footnote that already has one, which is worse than living with the
margins. *"Fourteen of the fifteen"* is accurate as written. If a reviewer
raises it, the answer is this table.

### A correction to the footnote, found while checking these rows

The footnote used to say *"three of the four families are independent
sources, since the FastV rows are reproductions run by others."* That is
wrong. VLA-Cache is a reproduction in both tables exactly as FastV is.

| family | appears in | its own authors' results |
|---|---|:-:|
| EfficientVLA | EfficientVLA Table 2 | ✓ |
| VLA-Pruner | VLA-Pruner Table 2 | ✓ |
| FastV | both | ✗ |
| VLA-Cache | both | ✗ |

Two of four, not three. The defensible claim is the one the footnote now
makes, that the fifteen rows come from **two independently authored tables**
and span **four method families**, and that neither paper remarks on the
split.

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
