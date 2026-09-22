# Narration, wall deck (19 slides, 178 s)

| # | s | words | w/s | narration |
|---|---|---|---|---|
| 1 | 13 | 33 | 2.5 | Vision-language-action models are capable but slow. Every call runs a vision encoder and a language backbone. Training-free tricks promise cheaper inference, but gains are reported for one backbone and benchmark. Do they transfer? |
| 2 | 11 | 28 | 2.5 | Trick one, visual foveation. It keeps a sharp central disc and blurs the periphery. Token count is unchanged, so it changes what the policy sees, not its cost. |
| 3 | 11 | 27 | 2.5 | Trick two, action repeat. Each action is held for two or four steps, so the policy is called less often. No state check, a pure speed baseline. |
| 4 | 12 | 30 | 2.5 | Trick three, depth pruning. Decoder blocks that barely change their hidden states are removed, one, two, or four, keeping the early and final blocks. The selection is fixed before evaluation. |
| 5 | 11 | 28 | 2.5 | Trick four, guarded reuse. The previous action is reused only while every gate passes: image change, action agreement, translation, and gripper. Any failed gate restores a full call. |
| 6 | 12 | 31 | 2.6 | Trick five, temporal fusion. Visual tokens of stable patches are reused from the previous call, while patches with motion, entropy, or language attention are recomputed. The policy still runs every step. |
| 7 | 8 | 21 | 2.6 | Now every configuration on one episode. CogACT on WidowX, eggplant: 12 of 14 configurations succeed, the two action repeat runs fail. |
| 8 | 6 | 9 | 1.5 | Stack cube, one episode: 11 of 14 configurations succeed. |
| 9 | 6 | 13 | 2.2 | Spoon on towel, one episode: the original fails, 6 of 14 configurations succeed. |
| 10 | 8 | 20 | 2.5 | OpenVLA on Fractal, again one episode per slide. Move near: 12 of 14 configurations succeed, the two foveation runs fail. |
| 11 | 8 | 10 | 1.2 | Pick coke can, one episode: 13 of 14 configurations succeed. |
| 12 | 8 | 12 | 1.5 | Open drawer, one episode: the original fails, 8 of 14 configurations succeed. |
| 13 | 6 | 19 | 3.2 | UniVLA on LIBERO, one episode per suite. Spatial: 12 of 14 configurations succeed, the two action repeat runs fail. |
| 14 | 6 | 8 | 1.3 | Object, one episode: 12 of 14 configurations succeed. |
| 15 | 6 | 8 | 1.3 | Goal, one episode: 12 of 14 configurations succeed. |
| 16 | 7 | 14 | 2.0 | Long, one episode: 12 of 14 configurations succeed, the two action repeat runs fail. |
| 17 | 13 | 31 | 2.4 | Across seven backbones and three environments, guarded reuse and temporal fusion are the safest for preserving success. Foveation and depth pruning help selected backbones. Action repeat is mainly a speed baseline. |
| 18 | 13 | 32 | 2.5 | Second, speed. Action repeat gives the largest speedup. SpatialVLA on WidowX drops from about 424 to 235 milliseconds per step. Depth pruning gives smaller decoder-level savings, and guarded reuse the safest trade-off. |
| 19 | 13 | 33 | 2.5 | The same trick helps one backbone and hurts another. Guarded reuse is most reliable, action repeat fastest but risky, depth pruning backbone-dependent. Evaluate every trick under the target backbone and environment. Thank you. |
