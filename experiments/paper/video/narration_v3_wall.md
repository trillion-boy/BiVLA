# Narration script, v3 wall deck

Total planned: 175 s over 17 slides (limit 180 s).

## Slide 1 (13 s, 36 words, 2.8 w/s)
Vision-language-action models are capable but slow. Every policy call runs a vision encoder and a language backbone. Training-free tricks promise cheaper inference, but gains are typically reported for one backbone and one benchmark. Do they transfer?

## Slide 2 (11 s, 27 words, 2.5 w/s)
Trick one, visual foveation. The observation keeps a sharp central disc and blurs the periphery. Token count and compute are unchanged. Only what the policy sees changes.

## Slide 3 (11 s, 29 words, 2.6 w/s)
Trick two, action repeat. Each predicted action is held for two or four control steps, so the policy is called less often. No state check, a pure speed baseline.

## Slide 4 (12 s, 32 words, 2.7 w/s)
Trick three, depth pruning. Decoder blocks that barely change their hidden states are removed, one, two, or four of them, keeping the early and final blocks. The selection is fixed before evaluation.

## Slide 5 (11 s, 28 words, 2.5 w/s)
Trick four, guarded reuse. The previous action is reused only while every gate passes: image change, action agreement, translation, and gripper. Any failed gate restores a full call.

## Slide 6 (12 s, 31 words, 2.6 w/s)
Trick five, temporal fusion. Visual tokens of stable patches are reused from the previous call, while patches with motion, entropy, or language attention are recomputed. The policy still runs every step.

## Slide 7 (12 s, 29 words, 2.4 w/s)
Seven backbones, official checkpoints, on WidowX, Fractal, and LIBERO. Thirteen trick settings per pair plus the original, 286 trick settings, each compared with the original on the same episodes.

## Slide 8 (6 s, 13 words, 2.2 w/s)
Now every configuration on one episode. CogACT on WidowX, 6 of 14 succeed.

## Slide 9 (8 s, 8 words, 1.0 w/s)
Eggplant in the basket: 11 of 14 succeed.

## Slide 10 (6 s, 8 words, 1.3 w/s)
Spoon on the towel: 11 of 14 succeed.

## Slide 11 (8 s, 12 words, 1.5 w/s)
OpenVLA on Fractal, same episode for every configuration. 9 of 12 succeed.

## Slide 12 (8 s, 9 words, 1.1 w/s)
Move the object near another: 9 of 12 succeed.

## Slide 13 (8 s, 8 words, 1.0 w/s)
Pick the coke can: 9 of 12 succeed.

## Slide 14 (12 s, 27 words, 2.2 w/s)
First, success. Guarded reuse and temporal fusion are the safest for preserving success. Foveation and depth pruning help selected backbones. Action repeat is mainly a speed baseline.

## Slide 15 (13 s, 32 words, 2.5 w/s)
Second, speed. Action repeat gives the largest speedup. SpatialVLA on WidowX drops from about 424 to 235 milliseconds per step. Depth pruning gives smaller decoder-level savings, and guarded reuse the safest trade-off.

## Slide 16 (12 s, 24 words, 2.0 w/s)
Third, consistency. Across backbone and environment pairs the same trick improves one model, leaves another unchanged, and hurts a third. No trick is plug-and-play.

## Slide 17 (12 s, 29 words, 2.4 w/s)
Guarded reuse is most reliable. Action repeat is fastest but risky. Depth pruning is backbone-dependent. Evaluate every trick under the target backbone and environment before applying it. Thank you.
