# Narration script for the ICRA 2027 accompanying video

Total: 165 s across 16 slides (limit 180 s).

## Slide 1: Title (7 s, 13 words, 1.9 w/s)

Narration: This video accompanies an anonymous submission on training-free efficiency tricks for vision-language-action models.

Production: Static title card. No names, affiliations, logos, or URLs anywhere in the video.

## Slide 2: Headline (10 s, 22 words, 2.2 w/s)

Narration: Our headline finding. The same training-free trick, on the same benchmark, helps one model and hurts two others. No trick is plug-and-play.

Production: Numbers animate in one by one (CogACT, then MiniVLA, then SpatialVLA). All three from Table I, WidowX, depth pruning row.

## Slide 3: Problem (10 s, 24 words, 2.4 w/s)

Narration: VLA models are capable but slow. Each control step runs a vision encoder and language backbone. Existing tricks are each reported in one setting.

Production: Left: recorded original-policy episode with a latency counter burned in. Right: static pipeline stack.

## Slide 4: Three axes (12 s, 30 words, 2.5 w/s)

Narration: We study five training-free tricks along three axes of the control loop. What to see, when to act, and how much to compute. Each trick changes both cost and behavior.

Production: Fig. 2(a) of the paper. Highlight each trick box in sequence as it is named. The original policy is the unmodified loop and counts as one of six variants.

## Slide 5: What to see (8 s, 19 words, 2.4 w/s)

Narration: What to see. Visual foveation keeps a sharp central disc and blurs the periphery. The token count is unchanged.

Production: The three frames are a WidowX observation rendered with the foveation mixture at keep ratio 20% and 50% (replace with frames from the actual foveation code if preferred).

## Slide 6: When to act (10 s, 24 words, 2.4 w/s)

Narration: When to act. Action repeat holds each action for two or four steps. Guarded reuse skips the policy call only when every gate passes.

Production: Timeline of policy calls. Optionally animate the dots left to right in step with the narration.

## Slide 7: How much to compute (10 s, 25 words, 2.5 w/s)

Narration: How much to compute. Depth pruning removes one, two, or four decoder layers with lowest Block Influence. Temporal fusion reuses visual tokens of stable patches.

Production: Two crops from Fig. 2(a). Optional: fade out the pruned layers in the left diagram.

## Slide 8: Protocol (12 s, 28 words, 2.3 w/s)

Narration: Seven open-source backbones, official checkpoints, no retraining, on SimplerEnv WidowX, Fractal, and LIBERO. Thirteen trick settings per pair, 286 settings in total, same initial states, paired McNemar tests.

Production: 286 = 13 trick settings × 22 pairs (the original policy is the 14th configuration of each pair).

## Slide 9: RQ-1 success (12 s, 28 words, 2.3 w/s)

Narration: First, success. Guarded reuse and temporal fusion match or improve success for most backbones. Foveation and depth pruning help selected backbones. Action repeat is mainly a speed baseline.

Production: Optionally enlarge one row of the 2×3 panel at a time. Outline guarded-reuse and temporal-fusion bars in green.

## Slide 10: Qualitative CogACT (12 s, 25 words, 2.1 w/s)

Narration: CogACT on WidowX. In these paired episodes the original fails and depth pruning succeeds. Overall success rises from 50.0 to 59.5, and latency drops slightly.

Production: Fig. 3(a) strips (Project Page mention cropped off). Bottom-left: side-by-side recorded episode, original left, depth pruning right, success/fail badge at the end.

## Slide 11: Same trick, different backbone (10 s, 24 words, 2.4 w/s)

Narration: Same trick, same benchmark, different backbone. On MiniVLA, depth pruning drops success by 17.5 points, on SpatialVLA by 6.5. Here the pruned policy fails.

Production: Record a MiniVLA WidowX episode from the same task family as slide 10 (e.g. carrot on plate) for original and depth pruning.

## Slide 12: RQ-2 speed (12 s, 25 words, 2.1 w/s)

Narration: Second, speed. Action repeat gives the largest speedup. SpatialVLA on WidowX drops from 423.5 to 235.1 milliseconds per step. Depth pruning gives smaller decoder-level savings.

Production: Fig. 4: speedup vs success change, star = original, shaded = faster and no worse. Highlight the action-repeat points at the far right and the depth-pruning points near the star.

## Slide 13: Repeat vs reuse in one episode (14 s, 35 words, 2.5 w/s)

Narration: Action repeat runs open loop. On WidowX, CogACT loses 38 points and UniVLA 75. Guarded reuse skips calls only when the gates agree. SmolVLA on LIBERO Long saves 20 milliseconds per step with success unchanged.

Production: Three-panel recording of one CogACT WidowX episode: original, action repeat k=4, guarded reuse. Under each panel a "policy calls used" counter; in the third panel a five-gate strip that turns green on skipped steps.

## Slide 14: RQ-3 consistency (12 s, 28 words, 2.3 w/s)

Narration: Third, consistency. Across backbone and environment pairs the same trick improves one model, leaves another unchanged, and hurts a third. Gains depend on the backbone and the environment.

Production: Consistency chart (Project Page figure). Do not read the counts aloud. Optionally highlight the depth-pruning and foveation rows.

## Slide 15: Takeaway (10 s, 27 words, 2.7 w/s)

Narration: Guarded reuse is most reliable. Action repeat is fastest but risky. Depth pruning is backbone-dependent. Evaluate every trick under the target backbone and environment before applying it.

Production: Four verdict cards in the paper's conclusion wording.

## Slide 16: End card (4 s, 7 words, 1.8 w/s)

Narration: Details are in the paper. Thank you.

Production: No URLs, names, or logos. Hold 4 s and fade to black.
