# Narration script, v2 (mentor structure: motivation, five tricks, results)

Total planned: 167 s over 14 slides (limit 180 s). Real timing follows the Speechma MP3 length per slide.

## Slide 1: MOTIVATION (15 s, 36 words, 2.4 w/s)

Narration: Vision-language-action models are capable but slow. Every control step runs a vision encoder and a language backbone. Training-free tricks promise cheaper inference, but each is reported on one model and one benchmark. Do the gains transfer?

Production: Static. Optional: play a short original-policy rollout behind the loop diagram. The three coloured markers on the loop show where foveation (orange diamond), temporal fusion (green dot) and depth pruning (purple dot) enter; they are explained on the trick slides.

## Slide 2: FOVEATION (11 s, 27 words, 2.5 w/s)

Narration: Trick one, visual foveation. The observation keeps a sharp central disc and blurs the periphery. Token count and compute are unchanged. Only what the policy sees changes.

Production: Left: Fig. 2 crop and example frames at keep 20% and 50%. Right: side-by-side rollout clip.

## Slide 3: ACTION REPEAT (11 s, 29 words, 2.6 w/s)

Narration: Trick two, action repeat. Each predicted action is held for two or four control steps, so the policy is called less often. No state check, a pure speed baseline.

Production: Timeline of policy calls (optional animation). Right: side-by-side rollout clip.

## Slide 4: DEPTH PRUNING (11 s, 26 words, 2.4 w/s)

Narration: Trick three, depth pruning. Decoder blocks that barely change their hidden states are removed, one, two, or four of them. The selection is fixed before evaluation.

Production: Left: Fig. 2 crop. Right: side-by-side rollout clip.

## Slide 5: GUARDED REUSE (11 s, 28 words, 2.5 w/s)

Narration: Trick four, guarded reuse. The previous action is reused only while every gate passes: image change, action agreement, translation, and gripper. Any failed gate restores a full call.

Production: Left: Fig. 2 crop and gated timeline. Right: side-by-side rollout clip with gate overlay.

## Slide 6: TEMPORAL FUSION (12 s, 31 words, 2.6 w/s)

Narration: Trick five, temporal fusion. Visual tokens of stable patches are reused from the previous call, while patches with motion, entropy, or language attention are recomputed. The policy still runs every step.

Production: Left: Fig. 2 crop. Right: side-by-side rollout clip.

## Slide 7: PROTOCOL (12 s, 29 words, 2.4 w/s)

Narration: Seven backbones, official checkpoints, on WidowX, Fractal, and LIBERO. Thirteen trick settings per pair plus the original, 286 trick settings, each compared with the original on the same episodes.

Production: 286 = 13 trick settings × 22 pairs.

## Slide 8: Q1 SUCCESS (12 s, 27 words, 2.2 w/s)

Narration: First, success. Guarded reuse and temporal fusion are the safest for preserving success. Foveation and depth pruning help selected backbones. Action repeat is mainly a speed baseline.

Production: Fig. 1 with a native legend. Optionally enlarge one row of panels at a time.

## Slide 9: QUALITATIVE (12 s, 25 words, 2.1 w/s)

Narration: CogACT on WidowX. In these paired episodes the original fails and depth pruning succeeds. Overall success rises from 50.0 to 59.5, and latency drops slightly.

Production: Fig. 3(a) strips. Bottom-left: side-by-side recorded episode.

## Slide 10: SAME TRICK, DIFFERENT BACKBONE (10 s, 24 words, 2.4 w/s)

Narration: Same trick, same benchmark, different backbone. On MiniVLA, depth pruning drops success by 17.5 points, on SpatialVLA by 6.5. Here the pruned policy fails.

Production: Record a MiniVLA WidowX episode from the same task family as the CogACT clip.

## Slide 11: Q2 SPEED (13 s, 32 words, 2.5 w/s)

Narration: Second, speed. Action repeat gives the largest speedup. SpatialVLA on WidowX drops from about 424 to 235 milliseconds per step. Depth pruning gives smaller decoder-level savings, and guarded reuse the safest trade-off.

Production: Fig. 4. If the Speechma clip runs longer than 13 s, extend the slide rather than speeding the voice.

## Slide 12: OPEN LOOP VS GATING (13 s, 35 words, 2.7 w/s)

Narration: Action repeat runs open loop: on WidowX, CogACT loses 38 points and UniVLA 75. Guarded reuse skips calls only when the gates agree. On SmolVLA in LIBERO Long, success stays at 42.0 with lower latency.

Production: Three-panel recording of one CogACT WidowX episode. Before rendering, fill the three "policy calls used" counters from that episode: original = number of steps, k = 4 = about steps / 4, guarded = steps minus skipped calls.

## Slide 13: Q3 TRANSFER (12 s, 24 words, 2.0 w/s)

Narration: Third, consistency. Across backbone and environment pairs the same trick improves one model, leaves another unchanged, and hurts a third. No trick is plug-and-play.

Production: Consistency chart (Project Page figure, counts from the per-setting records with LIBERO episodes pooled). Do not read the counts aloud.

## Slide 14: TAKEAWAYS (12 s, 29 words, 2.4 w/s)

Narration: Guarded reuse is most reliable. Action repeat is fastest but risky. Depth pruning is backbone-dependent. Evaluate every trick under the target backbone and environment before applying it. Thank you.

Production: Final slide. No names, logos, or URLs. Hold 2 s after the narration ends, then fade to black.
