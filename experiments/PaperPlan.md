
### What must be written in final form now, because the five are going in

Asked 2026-08-22 once the mentor confirmed the five models join *this* paper
rather than a follow-up. The distinction that matters:

> **Numbers can wait. Methodological claims cannot.** A figure gets updated
> when the runs land. A claim about how we measure, published in a form the
> final grid contradicts, is a defect in the paper.

**One sentence needed fixing, contribution 4.** It read *"require every cell to
run the same conditions."* That is already imprecise on the current grid. The
five cells share an identical **core eight**, but their full condition sets
differ, since `depth_prune8` ran in two cells, `prune4_back` in two,
`prune4_early` and `prune4_mid` in one each, and `prune2_repeat2` in one. On
the expanded grid it breaks outright, because depth pruning applies to two of
the five new models.

It now reads *"hold the condition set fixed across the cells we compare."*
That is the rule the comparisons actually rest on, it is true of the current
grid, and it stays true however ragged the grid becomes. A cell that cannot
run a condition is then reported as inapplicable with its reason, which is
`FiveModels_Read.md` §2's finding rather than an excuse.

**Everything else that changes is a number**, and the update is mechanical.

| where | now | after the expansion |
|---|---|---|
| setup paragraph | "three open backbones", "two SimplerEnv suites", "five of the six cells", "eight conditions", "$7{,}198$ episodes" | new counts |
| result 1, close | "Across all five cells the contrast spans $2.1$ to $50.4$" | recount, only where depth pruning applies |
| result 3, close | "ten of the fourteen intervention conditions our two Fractal cells ran" | FLOWER is the only new Fractal cell |
| contribution 1 | "$3 \times 2$ grid, five filled cells with eight conditions each" | new counts |
| closing | "A broken setup would push all five down" | "all of them" |

**One sentence will need rewriting rather than renumbering**, and that is fine
to leave until the runs land. The setup paragraph explains the empty cell with
a single named reason, *"UniVLA releases a Bridge-only checkpoint."* With
eight backbones there are several reasons for several gaps, so that clause
becomes a sentence about which cells exist and why. Writing it the current way
now is correct and clear, and rewriting it later costs one sentence.
