# Narration, wall deck (19 slides, 178 s)

| # | s | words | w/s | narration |
|---|---|---|---|---|
| 1 | 13 | 36 | 2.8 | Vision-language-action models are capable but slow. Every policy call runs a vision encoder and a language backbone. Training-free tricks promise cheaper inference, but gains are typically reported for one backbone and one benchmark. Do they transfer? |
| 2 | 11 | 32 | 2.9 | Trick one, visual foveation. The observation keeps a sharp central disc and blurs the periphery. The token count is unchanged, so this changes what the policy sees, not how much it computes. |
| 3 | 11 | 29 | 2.6 | Trick two, action repeat. Each predicted action is held for two or four control steps, so the policy is called less often. No state check, a pure speed baseline. |
| 4 | 12 | 32 | 2.7 | Trick three, depth pruning. Decoder blocks that barely change their hidden states are removed, one, two, or four of them, keeping the early and final blocks. The selection is fixed before evaluation. |
| 5 | 11 | 28 | 2.5 | Trick four, guarded reuse. The previous action is reused only while every gate passes: image change, action agreement, translation, and gripper. Any failed gate restores a full call. |
| 6 | 12 | 31 | 2.6 | Trick five, temporal fusion. Visual tokens of stable patches are reused from the previous call, while patches with motion, entropy, or language attention are recomputed. The policy still runs every step. |
| 7 | 8 | 13 | 1.6 | Now every configuration on one episode. CogACT on WidowX, 12 of 14 succeed. |
| 8 | 6 | 6 | 1.0 | Stack cube: 11 of 14 succeed. |
| 9 | 6 | 11 | 1.8 | Spoon on towel: the original fails, 6 of 14 configurations succeed. |
| 10 | 8 | 17 | 2.1 | OpenVLA on Fractal, same episode for every configuration. Move near: 12 of 14 succeed, only foveation fails. |
| 11 | 8 | 7 | 0.9 | Pick coke can: 13 of 14 succeed. |
| 12 | 8 | 10 | 1.2 | Open drawer: the original fails, 8 of 14 configurations succeed. |
| 13 | 6 | 15 | 2.5 | UniVLA on LIBERO, four suites, same episode for every configuration. Spatial: 12 of 14 succeed. |
| 14 | 6 | 5 | 0.8 | Object: 12 of 14 succeed. |
| 15 | 6 | 5 | 0.8 | Goal: 12 of 14 succeed. |
| 16 | 7 | 9 | 1.3 | Long: 12 of 14 succeed. Only action repeat fails. |
| 17 | 13 | 31 | 2.4 | Across seven backbones and three environments, guarded reuse and temporal fusion are the safest for preserving success. Foveation and depth pruning help selected backbones. Action repeat is mainly a speed baseline. |
| 18 | 13 | 32 | 2.5 | Second, speed. Action repeat gives the largest speedup. SpatialVLA on WidowX drops from about 424 to 235 milliseconds per step. Depth pruning gives smaller decoder-level savings, and guarded reuse the safest trade-off. |
| 19 | 13 | 37 | 2.8 | The same trick helps one backbone, leaves another unchanged, and hurts a third. Guarded reuse is most reliable, action repeat fastest but risky, depth pruning backbone-dependent. Evaluate every trick under the target backbone and environment. Thank you. |
