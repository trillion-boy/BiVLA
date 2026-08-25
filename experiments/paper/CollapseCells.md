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

## 4. How action repeat is applied to a chunked policy

Asked because a reviewer cannot check the mechanism without it, and because
"20 environment steps" is only true under one of three plausible readings.

`eval.py:582` is the whole answer, and its own comment states the choice:

```python
# Repeat each element in place (np.repeat semantics), NOT tile -- [a,b] -> [a,a,b,b].
env_actions = [a for a in env_actions for _ in range(args.action_repeat)]
```

So each action in the predicted chunk is **held for `repeat` consecutive
environment steps**. Not the chunk tiled `repeat` times, and not the same chunk
executed once with the policy simply called less often. For UniVLA at repeat 4
the executed sequence is $a_1a_1a_1a_1\,a_2a_2a_2a_2\ldots a_5a_5a_5a_5$,
twenty steps, all of them open loop.

**And the harness already computed this.** `paired_test.py` has a `horizon()`
function whose docstring says the thing a reviewer would raise:

> *"Env steps executed per model call... This is the quantity the temporal
> conditions actually change, and it is not comparable across backbones
> without being stated: a chunking policy at action-repeat 2 sits at 2x its
> chunk length, not at 2."*

Read straight out of the result files, not reconstructed:

| backbone | repeat | chunk | **horizon** |
|---|---:|---:|---:|
| OpenVLA | 4 | 1 | **4** |
| SpatialVLA | 4 | 1 | **4** |
| UniVLA | 2 | 5 | **10** |
| **UniVLA** | **4** | **5** | **20** |

## 5. The strawman charge, and the honest answer to it

*"Twenty steps of open loop is a setup built to fail. Reporting that as a
result is a strawman."*

**Half of that is right, and it decides how the row may be reported.**

What we may **not** write is that action repetition is worse for UniVLA than
for the other two. At repeat 4 the three backbones are not running the same
intervention, so a comparison down that column would be exactly the error this
paper is about, committed by us.

What we **may** write, and what the row is for, is that a single nominal
setting produced three different operations. `Action repeat = 4` is 4
environment steps on two backbones and 20 on the third, and a paper reporting
the setting without the chunk length has told the reader nothing about which
one it ran. That is result 1 on the temporal axis rather than the depth axis.

**One fact makes the charge weaker than it looks: the horizons cannot be
matched.** UniVLA's chunk is 5, so the horizons available to it are 5, 10, 15,
20. There is no repeat value that puts it at OpenVLA's 4. Matching the nominal
setting and matching the horizon are not both possible here, and any uniform
protocol has to pick one. We picked the nominal setting, which is what the
literature reports, and the cost of that choice is the finding.

**What this obliges Setup to state**, in one sentence each: the repeat
semantics above, the three native chunk lengths, and that the horizons are
therefore 4 / 4 / 20 rather than 4 / 4 / 4.

## 6. What Results has to carry

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
