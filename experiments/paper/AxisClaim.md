# The axis claim, and why it is not an Introduction result

Written 2026-08-22 during a reviewer pass. Recorded because the claim was a
headline for weeks and the arithmetic that kills it takes one line.

## What the Introduction used to say

> **In our grid the backbone matters more than the benchmark.** One of the
> fourteen benchmark-axis comparisons survives correction, against six of the
> twenty-nine along the backbone axis.

Both counts are correct. The Fisher interaction family has $43$ tests at
$\alpha = 0.05/43 \approx 0.0012$, and they split $14$ along the benchmark
axis and $29$ along the backbone axis, with $1$ and $6$ survivors.

## Why it cannot headline

$1/14$ against $6/29$ is $7.1\%$ against $20.7\%$. Run the same test on those
counts that the paper runs on everything else:

| | survives | does not | |
|---|---:|---:|---|
| benchmark axis | 1 | 13 | |
| backbone axis | 6 | 23 | Fisher $p = 0.396$ |

**The two rates are not distinguishable in our grid.** A paper whose fourth
contribution is that the field states differences without testing them cannot
open with a difference it did not test. That is the single most quotable
objection a reviewer could raise against us, and it would be correct.

The counts are also structurally unequal for a reason that has nothing to do
with the effect. The benchmark axis needs a backbone present on both suites,
which only OpenVLA and SpatialVLA are, so it gets $2$ pairs. The backbone axis
gets $3$ pairs on Bridge and $1$ on Fractal. The benchmark axis has half the
tests because of which cells exist, not because of what is in them.

## What replaced it

> **The same intervention reverses sign across backbones.** On the same $135$
> Fractal episodes under the same intervention, OpenVLA gains $+15.6$ while
> SpatialVLA loses $-17.8$, each surviving correction on its own and differing
> from the other.

Every part of that is tested. $+15.6$ at McNemar $p = 1.07\times10^{-3}$,
$-17.8$ at $p = 1.82\times10^{-4}$, both under $\alpha = 0.05/38 \approx
0.0013$, and the two cells differ at Fisher $p = 4.30\times10^{-7}$. It is
also the more striking claim, since a sign reversal is stronger than a rate
comparison.

## What Results should do with the counts

Report them, because they are real and they are the shape of the grid. Report
the Fisher $p = 0.396$ next to them and say plainly that the grid is too small
to separate the axes. Six cells with one empty is not enough, and saying so is
a limitation the reader will trust us more for.

**Do not let the counts drift back into a claim.** The sentence to avoid is
any form of *"the backbone matters more than the benchmark."* The sentence
that is safe is *"more of our surviving differences lie along the backbone
axis, though the grid cannot establish that the axes differ."*
