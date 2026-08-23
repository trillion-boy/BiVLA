# Two objections to result 3, and what the repository can answer

Written 2026-08-22 from reviewer-style objections to *"the same intervention
reverses sign across backbones."* Both are aimed at Setup, Results and
Limitations rather than at the Introduction, which does not have to answer
them. The first has a defence in our own data. The second does not, and
should be conceded.

---

## Objection 1. Four layers is not the same dose for both backbones

> *"OpenVLA and SpatialVLA do not have the same number of transformer layers.
> Removing four from each takes a larger fraction from the smaller one, so
> SpatialVLA collapsing may be your harsher treatment rather than a property
> of the backbone."*

### The objection is correct as stated, and the paper does not currently say so

Layer counts, recovered from the `redundancy` array each pruning run logs and
confirmed by `Report.md` §(that table at line 623):

| backbone | layers | four layers is |
|---|---:|---:|
| OpenVLA | 32 | $12.5\%$ |
| UniVLA | 32 | $12.5\%$ |
| **SpatialVLA** | **26** | **$15.4\%$** |

So the doses differ, by $2.9$ percentage points. Nothing in the Introduction,
and nothing yet drafted for Setup, tells the reader this.

*(A reviewer guessing 32 against 16 would expect $12.5\%$ against $25\%$. The
real gap is much smaller, but it is not zero and it has to be disclosed.)*

### Our own data already rules the fraction hypothesis out

`Report.md` line 621 onward, the back-half capacity ladder on Bridge:

| backbone (Bridge) | deleting only from the back half |
|---|---|
| OpenVLA (32 layers) | **8 layers removed, $\pm 0.0$** |
| UniVLA (32 layers) | 8 layers removed, $-4.2$ ($p = 0.42$) |
| **SpatialVLA (26 layers)** | **4 layers removed, $-30.2$** (29 broken, 0 fixed) |

**OpenVLA absorbs eight layers, $25\%$ of its stack, at zero cost.** That is a
strictly more aggressive treatment than the $15.4\%$ that costs SpatialVLA
$30.2$ points. If the fraction were the operative variable, OpenVLA at $25\%$
would have to be worse than SpatialVLA at $15.4\%$, and it is not.

`Report.md` draws the same conclusion in place: *"SpatialVLA has no spare
layers. For the other two the back eight are effectively free."*

### The sentence Setup or Results should carry

> We hold the number of removed layers fixed rather than the fraction, since
> OpenVLA and UniVLA have 32 decoder layers and SpatialVLA has 26, so four
> layers is $12.5\%$ against $15.4\%$. The difference does not drive the
> reversal. OpenVLA gives up eight layers, $25\%$ of its stack, for $\pm 0.0$,
> a heavier dose than the one that costs SpatialVLA $30.2$ points.

**Do not** claim we controlled the fraction. We did not. Claim that the
fraction hypothesis is refuted by a condition we ran anyway.

---

## Objection 2. You show the phenomenon but not the cause

> *"Fine, the sign reverses. **Why?** What is structurally different between
> these backbones that makes one survive and the other die?"*

### We do not know, and the paper must not pretend otherwise

What the repository does **not** contain:

- no attention-map visualisation for any backbone
- no measurement linking the architecture difference (SigLIP plus DINOv2
  against PaliGemma 2) to the reversal
- no pretraining-mixture analysis that would separate data from architecture

The `move_near` mechanism probe, 240 episodes, asks **which task breaks and
how**, not **which backbone differs and why**. It cannot be repurposed into a
causal account.

### That is the paper's design, not a hole in it

Contribution 2 claims that a reported effect **can be** a property of the
configuration. It claims existence and magnitude, not mechanism. The closing
paragraph draws the same line, *"We propose no new efficiency method ... Our
claim is narrower."*

The one thing we can say about mechanism is descriptive rather than causal,
and it is in the table above: **SpatialVLA has no spare depth where the other
two do.** That is a restatement of the measurement at a slightly higher level,
not an explanation of why its depth is not spare.

### The sentence Limitations should carry

> We establish that the direction of a depth-pruning result depends on the
> backbone, and we do not explain why. Differences in attention structure or
> in pretraining mixture are candidates, and separating them needs
> instrumentation this grid does not carry. What the paper claims is that the
> cause, whatever it is, is already inside results the field reports without
> it.

### Why conceding is the stronger move

A five-cell grid cannot support a mechanism claim, and a reviewer who finds
one will attack it as unsupported. A reviewer who finds the concession finds
an author who knows the boundary of their evidence. The concession also costs
nothing, because the paper's contribution is the measurement procedure and the
demonstration, neither of which needs the mechanism.

**The one real risk is silence.** A reviewer primed by result 3 will look for
the *why* in Discussion, fail to find it, and write that the paper does not
explain its own headline. Stating the limit up front converts that from a
criticism into a scope statement.

---

## What neither objection changes

The Introduction stays as written. It reports the reversal, the two p-values,
and the cell average that hides it. It promises no mechanism and claims no
fraction control, so nothing in it becomes false when Setup and Limitations
add these two passages.
