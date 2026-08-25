# The three cells past the clip, and why none of them is a bug

Written 2026-08-25, after a reviewer read Table I and asked whether
UniVLA/Bridge at $-81.2$ against an $81.2\%$ baseline meant the run had failed
outright. It is a fair question. A delta that cancels its own baseline to
exactly zero is what a broken harness produces.

**It is not a bug, and the records say so three separate ways.**

## 1. The number

| condition | successes | rate | paired delta |
|---|---:|---:|---:|
| baseline | 78 / 96 | 81.2\% | |
| action repetition 2 | 11 / 96 | 11.5\% | $-69.8$ |
| **action repetition 4** | **0 / 96** | **0.0\%** | $-81.2$ |
| foveation, log-polar | 83 / 96 | 86.5\% | $+5.2$ |
| depth pruning 4 | 76 / 96 | 79.2\% | $-2.1$ |

Zero of ninety-six, and zero on every one of the four tasks separately,
including `put_eggplant_in_basket`, which the baseline solves 24 out of 24.

## 2. Why a harness fault does not fit

**The degradation is graded, not a cliff.** Repeat 2 gives $11.5\%$ and repeat
4 gives $0\%$. A harness that scored every episode as a failure would give
$0\%$ at both. The intermediate value is the thing a bug cannot produce.

**The same cell is healthy under the other six conditions.** Foveation on the
same backbone, same benchmark, same episodes returns $+5.2$, and depth pruning
4 returns $-2.1$. Whatever breaks under action repetition is specific to
action repetition.

**The other backbones do not collapse under it.** OpenVLA/Bridge is $-11.5$
and SpatialVLA/Bridge is $-12.5$ at repeat 4. If our repeat implementation
were scoring episodes wrongly it would do so everywhere.

## 3. The mechanism, which is our own first result again

`FiveModels_Read.md` records the native chunk lengths of the three backbones
as **1 / 1 / 5**. OpenVLA and SpatialVLA emit one action per forward pass.
**UniVLA emits five.**

So "action repetition 4" is not one operation applied to three models. It is

| backbone | native chunk | $\times$ repeat 4 | environment steps per policy call |
|---|---:|---:|---:|
| OpenVLA | 1 | 4 | **4** |
| SpatialVLA | 1 | 4 | **4** |
| UniVLA | 5 | 4 | **20** |

Twenty environment steps between decisions on a manipulation task, from a
policy trained to decide every five. The collapse is what the arithmetic
predicts, and repeat 2 at ten steps landing on $11.5\%$ is the midpoint of the
same curve.

**This is result 1 in another costume.** The same named intervention means
different amounts of the same resource depending on a property of the
backbone, and no paper reporting "action repeat = 4" states the chunk length
that makes the number mean anything. `FiveModels_Read.md` already flags this
for the expansion, where the five new models run 8, 20--50 and 50.

## 4. What Results has to carry

A reader reaching $-81.2$ will stop, so the explanation must be adjacent to
it, not in an appendix. The three parts, in order of what a sceptic needs:

1. the raw counts, 0 / 96, and 0 / 24 on all four tasks
2. repeat 2 at $11.5\%$, which rules out the harness
3. the chunk length, 5, which supplies the mechanism

The other two clipped cells need less. UniVLA/Bridge repeat 2 at $-69.8$ is
the same story one step earlier, and SpatialVLA/Fractal repeat 4 at $-40.0$
sits at chunk 1, so it is a large but ordinary degradation.

⚠️ The Introduction does not mention any of this and should not. It cites
action repetition only as one of the three axes, and result 3 is about depth
pruning. Adding a collapse story there would spend the space result 1 needs.
