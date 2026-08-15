# Prior Work — What the Three Interventions Are, and Who Has Tried Them

> **This document is §2 and Appendix A, split out of `Report.md`.** The section
> numbers were kept, so when another document says "§2.5" or "Appendix A.3,"
> those pointers still work.
>
> | Document | What |
> |---|---|
> | `Overview_EN.md` | the whole project in 10 minutes (start here if you are new) |
> | `Report_EN.md` | experiment setup, results, open items — what *we* measured |
> | **This one** | **prior work — what *others* measured, and what we read in their tables** |

---

## Two ways to read this document

**The fast way** — read §2.1 for why these three axes, then **jump to §2.5.**
That is what we learned from reading the papers.

**The thorough way** — §2.2–2.4 answer the same five questions for each axis:
**(a) where it came from → (b) what it does → (c) why it was brought to VLA →
(d) what has already been tried on VLA → (e) what we look at that is new.**

Appendix A is a set of notes, one paper at a time, organized around seven
questions.

> **How far we checked.** For the seven papers in Appendix A (EfficientVLA,
> VLA-Cache, ShortGPT, MoLe-VLA, Gaze-Reg, Look Focus Act, Segment This
> Thing), we opened the original PDFs and checked every number, table, and
> sentence we quote. **Every value we copied from a table matched the
> original.** What was wrong was not values but **scope and attribution**, and
> there were four such errors — we described three papers' experimental scope
> too narrowly (Gaze-Reg, VLA-Cache, MoLe-VLA), we had **Gaze-Reg's backbones
> reversed** (the main backbone is Pi-0, we had written OpenVLA), we quoted
> Look Focus Act's success-rate gains **without their conditions**, and we
> assumed ShortGPT's **calibration data amount** was the same as ours. All
> four are fixed and logged in `Report.md` §7.1.
>
> Separately from our own errors, we found **three internal inconsistencies in
> the ShortGPT paper itself** (A.3) — those are not our mistakes.
>
> The last two papers are **not VLA papers — they are segmentation and
> imitation learning** — and we use them only in §2.3 (d), to measure when
> savings on the vision side turn into actual wall-clock time (A.6).
>
> We also checked the titles and sources of the background references cited in
> §2.2 and §2.3 (a) — Braylan et al. (AAAI-15 workshop), Dynamic Frame skip
> DQN (1605.05365), the frame-skip analysis (2102.03718), Traver & Bernardino
> (*Robotics and Autonomous Systems* 58(4), 2010), Schwartz (1977/1980), and
> five recent chunking papers (2511.19433, 2606.18589, 2604.02965, 2607.01804,
> 2603.28565). For these we confirmed **titles and main claims only** — we did
> not check numbers inside their tables the way we did for Appendix A.

---

# §2. Prior Work

> This section is written to stand on its own. For each intervention it fills
> the same five slots — **(a) origin → (b) what it does → (c) why it was
> brought to VLA → (d) what has been tried on VLA → (e) what we look at that
> is new.** Slot (b) is explanation, but it is not the goal — it is the setup
> for (c). Each section closes with (e), so the explanation always leads to
> what we actually do.

## 2.1 Why these three

Among the places where a VLA policy's compute can be cut, **the three we
picked** are:

| Axis | What it reduces | Our intervention |
|---|---|---|
| **Time** | **how often** the policy is called | action repeat k |
| **Vision** | **what the policy is shown** | foveation (log-polar / blur) |
| **Compute** | **how much of the network** runs per call | depth pruning |

On top of these, the **original policy** (no intervention) is the fourth
condition — the control that every change is measured against.

**This is not an exhaustive list.** Quantization, KV caching, early exit, and
speculative decoding serve the same goal, and none of them fits cleanly into
these three. We picked these three because **each touches a different
resource** — there is no reason for one axis's result to predict another's, so
if all three move the same way we can read that as a property of the backbone
or benchmark, and if they move differently, as a property of the intervention.
That distinction is the basis of §5.

> ⚠️ That the three axes are actually independent is a **design assumption,
> not something we tested.** The grid has exactly one cell with stacked
> interventions (SpatialVLA/Fractal's prune 2 + repeat 2), and even that is
> **two axes, not three** — so we have no general grounds to rule out
> interactions.

## 2.2 The time axis — action repeat

### (a) Origin

Holding one action for several steps is **the oldest efficiency lever** in
this literature. DQN made frame skip = 4 the standard (Mnih et al., *Nature*
2015), and Braylan et al. (AAAI workshop 2015) were the first to observe
systematically that **the frame-skip value strongly shapes performance.** That
second observation is the direct ancestor of our claim — that the effect of
the hyperparameter k is unstable. Later work moved toward learning the repeat
count instead of fixing it (Dynamic Frame skip DQN, the FiGAR family; arXiv
1605.05365), and the reason frame skip helps was settled as **it shortens the
horizon the policy has to look ahead over** (arXiv 2102.03718).

In robot imitation learning the same idea reappears as **action chunking** —
ACT (Action Chunking with Transformers) and Diffusion Policy made it standard
in 2023.

### (b) What it does

**Action repeat k**: call the policy once, get one action, and hold it for k
environment steps. The observation changes during those steps, but the policy
does not see it.

> ⚠️ **Do not confuse this with action chunking.** Chunking predicts k actions
> and executes k of them. Repeat **predicts one action and holds it for k
> steps.** Repeat is the degenerate form of chunking (H=1, s=k). If we do not
> say this, a reviewer will immediately.

### (c) Why bring it to VLA — and why repeat rather than chunking

Three reasons; the third matters most.

**① It is not a trick we picked — it is the speed-up this field already uses
by default.** Modern VLAs really do use chunking. So testing this is testing
one of the field's standing assumptions. (Lowering weight precision
(quantization) or reusing computation (caching) also buys speed, but neither
changes what the policy sees or when it is called, so they are outside our
question.)

**② The degenerate form is cleaner to analyze.** If success drops under
repeat, the loss came from **acting on a stale observation** — not from a
different action-prediction structure. The cause is pinned to one thing.
Chunk execution mixes stale observations with **the model's multi-step
prediction quality.** At the same time, repeat gives **a lower bound on what
chunking could deliver.**

**③ In a cross-backbone design, chunk execution does not qualify.** The
premise of this work is applying *the same intervention* while changing only
the backbone — and chunk execution becomes **a different intervention on each
backbone**, because native chunk lengths differ.

| Backbone | Actions executed per forward | What `--exec-chunk 2` actually means | Measured |
|---|---|---|---|
| OpenVLA | 1 | **not applicable** — there is no chunk to cut | — |
| SpatialVLA | 1 | 1 → 2, an **increase** = speed-up | +13.6pp, 1.9× |
| UniVLA | 5 | 5 → 2, a **decrease** = **slowdown** | −12.5pp, **2.3× slower** |

The same flag speeds up one backbone, slows down another, and is not even
defined on a third. On UniVLA it did not save compute at all — ms per
env-step rose from 603 to 1,414. This looks similar to the `--depth-min-layer`
issue in §2.4 but is different in kind: that one is **two implementations
reading the same argument differently**, which disappears once you align
them; this one comes from **backbones having different native chunk lengths**,
which cannot be aligned.

> ⚠️ **The three numbers in the table above are off-grid runs.** Chunk
> execution is not one of our 8 conditions, so there are no result files under
> `results/`; the numbers come from the record of an early exploratory run
> (`ChunkExecFoveation_univla.md`). The SpatialVLA/Bridge baseline at that
> time was 32.3%, not the 30.2% of the current grid. **We cite these only as
> the reason we dropped this axis, never as performance numbers.**

Action repeat k, by contrast, **applies to any policy, and the saving is
roughly 1/k.** Our measurements confirm it — across all five cells, k=2 saves
−49.8 to −52.3% and k=4 saves −74.8 to −76.6%. In other words, **the amount
saved is pinned to 1/k nearly independently of backbone and benchmark.**
**Repeat is the only time-axis intervention that means the same thing on every
backbone.**

| | OpenVLA<br>Bridge | OpenVLA<br>Fractal | SpatialVLA<br>Bridge | SpatialVLA<br>Fractal | UniVLA<br>Bridge |
|---|---:|---:|---:|---:|---:|
| k = 2 | −49.8% | −49.9% | −50.6% | −52.3% | −52.3% |
| k = 4 | −74.8% | −75.0% | −75.1% | −75.9% | −76.6% |

> **That is why this axis gives the cleanest contrast.** The amount saved is
> fixed, so the only thing that differs between cells is **what you lose in
> exchange** — and that runs from +12.5 to −69.8 (see (e) below).

> ⚠️ **Ready for the pushback.** The objection will be "you measured something
> nobody deploys." The answer is ② and ③ — we are not recommending a
> deployment setting; we are **testing the stability of the sign**, and that
> purpose needs an intervention whose meaning is preserved across backbones.
> Comparing chunk-execution results across backbones does not even parse.

### (d) What has been tried on VLA

OpenVLA-OFT (2025) made chunking the speed standard for VLA with parallel
decoding + action chunking. Mixture of Horizons (arXiv 2511.19433) shows the
trade-off systematically: **looking further ahead per call captures the large
motion better but makes fine manipulation less precise.** And 2026 brought a
number of remedies for observations going stale during chunk execution —
correcting with a world model (DREAM-Chunk, 2606.18589), speculative
verification (2604.02965), adapting the look-ahead length to the situation
(VLA-Corrector, 2607.01804), streaming execution (2603.28565). **That so many
remedies exist is itself evidence the problem is real.**

### (e) What we look at that is new

k is usually **fixed as an implementation detail and never reported as an
ablation.** When it is reported, it is one backbone, one benchmark. **Whether
the sign of that trade-off survives a change of backbone or benchmark has, as
far as we could find, not been addressed anywhere.**

In our measurements it does not survive. **On the same Bridge benchmark,
action repeat 2 is +12.5 on SpatialVLA (p = 0.0428) and −69.8 on UniVLA
(p < 0.0001).**

## 2.3 The vision axis — foveation

### (a) Origin

Log-polar starts with Schwartz (1977 / 1980) — the finding that **the mapping
from the retina to the visual cortex is well described by a logarithm**
(`w = log(z + a)`). Why robot vision used it is laid out in the standard
review by Traver & Bernardino (*Robotics and Autonomous Systems*, 2010):
**reduce the amount of data while keeping resolution at the center.** That is
where our motivation comes from. The line of work that *learns* where to look
starts with Recurrent Models of Visual Attention (NeurIPS 2014); we **fix the
center**, so we cite that line as a contrast, not as an ancestor.

### (b) What it does

Keep only **20% of the pixel budget** of the observation, favoring the area
near the center. There are two variants.

- **log-polar**: `cv2.warpPolar` actually **moves** pixels, then warps them
  back. So *information loss* and *geometric distortion* are mixed together.
  **Even the dead center keeps only 39%**, but in exchange the periphery
  survives, roughly — a gentle, global simplification.
- **blur**: a spatially varying Gaussian, so **no pixel moves.** Out to
  r ≈ 0.35 (the radius of the sharp disc at keep 20%) the image is
  **bit-for-bit the original**, and beyond that it fades quickly — down to 1%
  past r > 0.7. The center is perfectly preserved and the periphery drops off
  over a short band.

Here is where each variant erases how much, side by side. We applied keep 20%
to a real Bridge observation (640×480) and measured **how much of the
original's fine detail survives as you move away from the center** (r=0 is
the center, r=1 the corner).

| Distance from center r | 0.0–0.1 | 0.2–0.3 | 0.4–0.5 | 0.5–0.7 | 0.7–1.0 |
|---|---:|---:|---:|---:|---:|
| log-polar | 39% | 7% | 2% | 7% | 21% |
| blur | **100%** | **100%** | 54% | 10% | 1% |

**The two distributions are opposites.** Blur **leaves the center alone**
(100%) and erases the outside. Log-polar **cuts from the center first** (39%)
and keeps more of the outside.

> **There is more than one way to measure "how much detail survives."** The
> table above uses the **Laplacian**, which counts even the finest texture.
> Measure the same image with **Sobel**, which mostly sees strong edges, and
> the dead center reads **68%**, not 39%.
>
> So "39% at dead center" is not an intrinsic property of the transform — it
> is the **Laplacian-based** value. Numbers like this should always be quoted
> together with how they were measured. The ordering, though, survives either
> way: blur protects the center more, log-polar keeps more of the periphery.
>
> The rightmost cell going back up (2% → 21%) is a by-product of the reverse
> warp, and **it also depends on resolution** — shrink the same image to
> 256×256 and the 21% rises to 39–55% (**it even depends on how you shrink**:
> INTER_LINEAR gives 39%, INTER_AREA gives 55%; the direction — it goes up —
> is the same either way).
> To reproduce: `experiments/measure_foveation_roundtrip.py`.

**Having two variants is a design point.** But **do not read their difference
as "the share due to geometric distortion."** The table above is why — blur
keeps the center at 100% and erases the periphery, while log-polar cuts the
center to 39% and keeps more periphery. They are not *a pair that removes the
same information with and without a warp*; they are **two conditions that
remove different amounts of information in different places.** Subtracting one
from the other does not leave the geometry component.

What the two variants *do* show is that **different conditions hide under the
same intervention name**, and in §5.3 which variant wins actually flips from
cell to cell. To isolate the share of geometric distortion you would need a
**warp/no-warp pair matched for information content**, and we did not run
that.

### (c) Why bring it to VLA

The hypothesis: **a robot needs high resolution only where it acts; for the
periphery, context is enough.**

The place this hypothesis attaches is a property all three backbones
**share.** Their vision encoders differ — OpenVLA uses DINOv2 + SigLIP,
SpatialVLA uses SigLIP with Ego3D position encoding, UniVLA uses the Emu3 VQ
tokenizer — but **all three cut the image into a uniform grid and turn it
into a fixed number of units.** Whether those units are continuous patch
embeddings or discrete tokens, the grid is uniform either way. So **an empty
patch of background gets the same budget as the patch where the gripper meets
the object.**

Manipulation tasks are spatially concentrated near the center, so the hope is
that redistributing that budget toward where it is needed gives the policy
more information through the same number of units.

> The fact that this property is shared across encoder types is what makes
> foveation usable as a **cross-backbone axis** — in contrast to chunk
> execution in §2.2 (c)③, which could not be. Because the intervention touches
> the observation image **before** the encoder, its meaning does not change
> with the backbone.

### (d) What has been tried on VLA

**On the success-rate side — the two closest papers report opposite
results.** Our nearest neighbor is **Gaze-Regularized Vision-Language-Action
Models for Robotic Manipulation** (Pani & Yang, HKU). They build a foveated
RGB centered on the peak of a gaze distribution (high resolution at the
center, downsampled/blurred outside) and feed it to a standard vision encoder
— practically the same operation as our blur variant. Appendix D.2 / Table 11
(LIBERO-Spatial, 30k steps):

| Task | baseline | foveated | Δ |
|---|---|---|---|
| Between plate and ramekin | 83.3 | 80.0 | −3.3 |
| Next to ramekin | 85.7 | 81.3 | −4.4 |
| Table center | 100 | 95.7 | −4.3 |
| On cookie box | 100 | 90.0 | −10.0 |
| In cabinet drawer | 80 | 65.3 | **−14.7** |
| On ramekin | 100 | 90.0 | −10.0 |
| Next to cookie box | 100 | 94.0 | −6.0 |
| On stove | 90 | 80.7 | −9.3 |
| Next to plate | 50 | 44.7 | −5.3 |
| On wooden cabinet | 70.3 | 63.3 | −7.0 |
| **Overall** | **85.9** | **78.5** | **−7.4** |

**All 10 tasks drop; no exception.** The authors' reading:
*"aggressively reducing peripheral detail removes useful contextual cues
(e.g., table geometry, supporting surfaces, or alternative grasps) that the
policy relies on for precise spatial reasoning"* — that is, **the periphery is
context the policy needs.**

> ⚠️ **Two conditions that must accompany any citation of this.**
> **①** Their foveation is applied **during training** (the section is titled
> "Foveated Vision *during Training*", 30k steps). We apply it at inference,
> with no training.
> **②** Their center is the **peak of human gaze.** Ours is fixed at the image
> center.
>
> Note that both conditions **favor foveation.** They trained on that input
> and aligned the center with human gaze — and still dropped 10 out of 10. Our
> setting is less favorable on both counts — no training, center fixed — yet
> we got +18.8. So the result in (e) below is not explained by this paper.

**But Look, Focus, Act** (Chuang et al., arXiv 2507.15833, 2025) **reports the
opposite.** They use foveated tokenization that **re-cuts the patches
themselves** around human gaze (dense at the center, sparse outside), and
report ViT compute down 94%, inference 3× faster, and **success going up on
some high-precision tasks.**

> **That gain depends on conditions, though.** It is clearest in the
> **simulation + no ViT pre-training** setting; MAE pre-training shrinks the
> gap; and **on the real robot, uniform tokenization wins three of the four
> cells** (A.6 ④). So the citation should read not "foveation raises success"
> but **"it can raise success under some conditions."** That the direction
> splits is itself the point, and that stands.

> **Where the two papers diverge is in *what they change*.** Gaze-Reg **kept
> the token count and only blurred pixels**, reducing information (10/10
> drop). Look, Focus, Act **re-laid the patches and reduced the token count
> itself** (324 → 20, Table II). That is why the latter actually gets faster —
> the 94% ViT reduction and the 3× speed-up follow from it.
>
> **Our foveation belongs to the first kind** — image size and token count
> both stay the same. That fits the ≈0% compute saving we measured in §4.3,
> and the flip side is what the two papers together imply: **to buy speed you
> must touch the token count.** We did not run that variant.

**On the speed side — there is already a measurement showing it does not
save.** VLA-Cache Table 2 (LIBERO, OpenVLA, RTX 4090):

| | FLOPs (T) | Latency (ms) |
|---|---|---|
| OpenVLA | 1.864 | 51.91 |
| + FastV | **1.864** (no change) | **53.28** (+2.6%) |
| + SparseVLM | 1.407 (−24.5%) | **83.39** (**+60.6%**) |

**Cutting FLOPs does not cut time — it can even add time.** The causes the
authors name: these methods *"target long output sequences, whereas VLA
models generate short action outputs (e.g., 7 tokens)"*, and they work within
a single frame, which can *"disrupt spatial fidelity"*. EfficientVLA points at
the same wall: visual token pruning helps at first, but *"its efficacy quickly
diminishes as the system becomes memory-bound by the LLM"*, and FastV gets
*"only a 1.21× speedup due to unaddressed memory bottlenecks"*.

**Yet there is a place where cutting vision clearly buys time — where nothing
repeats after the encoder.** The same foveated tokenization appears in three
settings, and **the gain shrinks at each step.**

| Where it was applied | What repeats **after** the vision encoder | Vision-stage gain | Whole-system gain |
|---|---|---|---|
| **Segment This Thing** (segmentation) | nothing — the mask decoder runs once | — | **42×** (SAM-H 572.7 → STT-L 13.7 ms) |
| **Look, Focus, Act** (imitation policy) | flow matching, **8 steps** | ViT 243.8 → 16.4 ms = **14.9×** | whole policy 334.7 → 87.9 ms = **3.8×** |
| **VLA** (us) | **12–26 action tokens** × the whole decoder | (our variants do not cut tokens) | **≈0%** |

The three rows are one lineage: Look, Focus, Act states directly that it took
Segment This Thing's tokenization and *"adapt[ed] it for robot learning."*

**The middle row is the key.** In Look, Focus, Act, the ViT alone gets 14.9×
faster (324 → 20 tokens, 1905.4 → 115.6 GFLOPs = **−93.9%**), but the whole
policy lands at 3.8×. The authors say why, in their own words:

> *"During inference, differences are smaller because the flow matching
> transformer runs multiple sampling steps (8 in our case) while image
> features are processed only once by the ViT."*

**The vision stage runs once; the stages after it run many times.** So what
you save on vision gets diluted in the whole. For them, the repeated part was
a small flow transformer running 8 times — and that alone turned 14.9× into
3.8×.

**In a VLA, the repeated part is much heavier.** The profile in §2.4 (c) gives
the number — emitting action tokens one at a time passes through **the entire
decoder 12–26 times**, and that is 70–75% of one control step. Image encoding,
by contrast, is 13.9% (SpatialVLA) and 6.1% (UniVLA). **Put Look, Focus,
Act's 94% saving on top of those shares and this is what you get:**

| | Encoding share | Cut encoding by 94% | Make the vision stage **zero** |
|---|---:|---:|---:|
| SpatialVLA | 13.9% | 903 → 786 ms (**1.15×**) | 903 → 718 ms (**1.26×**) |
| UniVLA | 6.1% | 1,362 → 1,284 ms (**1.06×**) | 1,362 → 1,107 ms (**1.23×**) |

**So in a VLA, even deleting the vision side entirely gets you less than
1.3×.** The right column sets encoding *and* prompt reading to zero — an upper
bound no vision-side technique can beat. The wall VLA-Cache measured with
SparseVLM (FLOPs down 24.5%, time up 60.6% — table above) and EfficientVLA's
*"memory-bound by the LLM"* are the same story in different words.

> **Our variants do not even enter that table.** Both log-polar and blur
> change pixel values while **keeping image size and token count the same**,
> so the encoding cost itself does not shrink. That is why §4.3 (a) measured
> ≈0% savings — not an implementation gap but **what the design dictates.**
> Our foveation should be read not as a speed technique but as **an
> intervention that changes what the policy sees.**

**Evidence that *what you keep* dominates.** In EfficientVLA, keeping tokens
at random drops 74.8 → **20.9**. So the outcome is dominated less by how much
the token budget shrinks than by **which tokens survive.**

### (e) What we look at that is new

Nobody has measured this with paired episodes, and nobody has crossed
backbone × benchmark.

**And the last table in (d) is one only we can fill in.** That foveation buys
42× in segmentation and 3.8× in an imitation policy is other people's
reporting; **what the number becomes in a VLA** nobody had written down. We
have the profile, so we can put the number in that empty slot — **even making
the vision stage free tops out at 1.26× / 1.23×.** This is not a report that
our intervention failed; it is **the first concrete ceiling for this axis on
VLA**, and it says what vision-axis papers should check first when they move
over.

And more importantly: **within what we have read, we could not find a report
of a gain like our +18.8 under our conditions.** Reports that foveation helps
do exist (Look, Focus, Act) — but that setting **cuts tokens, uses human gaze,
and trains on the foveated input.** Ours keeps the token count, fixes the
center, and does no training — and still gained +18.8. What we could not find
is a reported gain under *that* combination. (We write "could not find," not
"does not exist" — negative claims cannot be proven, and this needs one more
check before submission.) A hypothesis in the **opposite direction** from
Gaze-Reg's ("the periphery is needed context") is required — that in this
cell, the periphery was not help but **interference.** Read together with
EfficientVLA's random-keep result, the working hypothesis is *"on
Bridge/OpenVLA, the periphery was hurting rather than helping,"* and the low
15.6% baseline may be a hint. It is the most striking cell in our grid, and
worth space in the paper.

## 2.4 The compute axis — depth pruning

### (a) Origin

Our layer-selection score comes from **ShortGPT**'s Block Influence:

```
BI_i = 1 − E[ cos( X_i , X_{i+1} ) ]
```

**It measures how much a layer changes its input.** The `cos` compares what
goes into a layer with what comes out; BI is one minus that. So **the more
alike input and output are, the lower the BI** — and a low-BI layer barely
changes anything, so the assumption is that it can be dropped.

The deletion order can be stated two equivalent ways — **lowest BI first**
(ascending) is the same order as **highest `cos` first** (descending).
ShortGPT writes it the first way; our code
(`SpatialVLA/experiments/tome/depth_prune_gemma2.py`) writes it the second.

> ⚠️ **Our calibration differs from ShortGPT in two ways.**
>
> To find out which layers are idle, you must **pass something through the
> model.** That input is called calibration data. The `E[ ]` in the formula
> means **an average over those inputs** — and what you feed, and how much,
> is where we differ.
>
> | | What goes in | How much |
> |---|---|---|
> | ShortGPT | **generic text** (PG19) — unrelated to the evaluation task | a calibration set of **many passages** |
> | Us | **the first observation of that run** — the actual task's screen | **one forward pass of one frame** |
>
> **① What you look at differs** — task-unrelated text vs. the first frame of
> the task about to be performed. **② How much you look at differs** —
> ShortGPT averages across many samples; we average only across token
> positions within a single forward pass (`_sum`/`_cnt` in
> `depth_prune_gemma2.py`). **Both differences must be stated whenever this is
> cited.**
>
> One thing that is the same on both sides: **neither re-measures during the
> run.** Our code calibrates once per run behind a `calibrated` flag
> (`tome_spatialvla_eval.py:299`), and the layer set is fixed afterward. That
> the result files show different layers per task is because **each task is a
> separate process**, each calibrating on its own first frame — not because
> anything is re-measured mid-run.
>
> **The side with less data is us, and that is a limitation that cuts against
> us.** Whether a ranking chosen from one first frame represents the whole
> episode is something we did not test.

### (b) What it does

The decoder is a stack of layers (32 in OpenVLA and UniVLA, 26 in
SpatialVLA). We pick the k layers that **do the least work and skip them.**
Skipping does not delete the layer or change weights — at run time, **the
input is passed straight through as the output.** No training, no fine-tuning
— just a detour built at execution time.

"Does the least work" is judged by the BI in (a). But **not every layer is a
candidate** — early layers are risky to touch, so a **candidate window** is
set first: from where onward deletion is allowed. The argument that sets that
window is `--depth-min-layer`.

> ⚠️ **One caution — we ran into this ourselves.** Two implementations read
> that same argument **differently.**
>
> | Implementation | How it reads `--depth-min-layer` | Example |
> |---|---|---|
> | OpenVLA · UniVLA | as a **fraction** | `0.5` → the back half of 32 layers, i.e. candidates L16–31 |
> | SpatialVLA | as a **count** | `2` → from L2 on, i.e. candidates L2–25 |
>
> So **the same value produces completely different windows.** Compare two
> backbones without knowing this and it is no longer "the same experiment" —
> this produced, and then resolved, one apparent counterexample in §6.

### (c) Why bring it to VLA

**Because this is where VLA's slowness lives.** We profiled one control step
stage by stage ourselves (`docs/VISUAL_TOKENS_VS_LATENCY.md`).

| Stage | SpatialVLA | UniVLA |
|---|---:|---:|
| image encoding | 125 ms (13.9%) | 83 ms (6.1%) |
| reading the prompt once | 60 ms (6.6%) | 172 ms (12.6%) |
| **emitting action tokens one by one** | **677 ms (75.0%)** | **951 ms (69.8%)** |
| environment & Python | 41 ms (4.5%) | 157 ms (11.5%) |
| **total** | **903 ms** | **1,362 ms** |

**Two structurally different backbones give the same answer** — SpatialVLA is
SigLIP ViT + Gemma2, UniVLA is a VQ tokenizer + Emu3, and **both spend 70–75%
of the time emitting action tokens one at a time.** The reason is simple:
every control step has to produce the action token by token, and each token
passes through **the entire decoder once.**

**Two things follow.**

1. **Shrinking the image side cannot help much.** Encoding plus prompt
   reading together are 20.5% (SpatialVLA) and 18.7% (UniVLA). **That is the
   ceiling.** Every vision-side technique we tried hit it — ToMe came out at
   0.99× (no change), and temporal caching at stride 2 saved 4%, and that was
   all.
2. **Only cutting decoder depth touches the 75%.** Delete layers and each
   token's pass gets cheaper by that fraction; that repeats 12–26 times, so
   the saving turns into real time.

**So of the three axes, this is the one where "does the saving become actual
time" can be observed.** On the vision axis (§2.3) the saving was ≈0%; here,
deleting 4 layers really does give **−11.2% to −15.9%** across the five
cells.

> **The conditions behind this profile.** SpatialVLA is a 25-step average,
> UniVLA a 6-step average (eggplant task). It is pure model inference time
> under CUDA synchronization, and unlike the success-rate experiments, **the
> sample is small** — enough to establish each stage's share, not to argue
> over decimals.

### (d) What has been tried on VLA

**Here the literature says firmly: it works.**

| Evidence | Conditions |
|---|---|
| EfficientVLA: on SIMPLER, FLOPs down **to 28.9%** (a 71.1% cut), average **−0.6 points** | **no training needed**, **the same score as ours**, **the same benchmark as ours** |
| EfficientVLA: pick coke can goes **91.3 → 93.3–95.3** (pruning *raises* success) | all four configurations. Table 2 |
| MoLe-VLA: skips half the layers and gains **+10.2 points** (over OpenVLA) — but **+3.6 points** over CogAct | the layer selector must be **trained**; one benchmark; and most of the gain comes from distillation (A.4) |
| ShortGPT: Llama2-13B, 25% of layers removed, MMLU **55.00 → 54.69** (Table 2) | LLM only — not a setting where **actions change the next frame**, as in robotics |

Our OpenVLA result (depth prune 4 at **+15.6** on Fractal; its interpretation
is left open in §4.4 c) points the same way as this literature.

### (e) What we look at that is new

**A second backbone.** The positive results we saw mostly live inside one
policy family. Our SpatialVLA drops substantially at 4 layers in both cells —
**Bridge −28.1, Fractal −17.8** — the opposite of the "nearly free" the
literature reports.

> But **at 1 layer the two cells split** — Bridge −10.4, Fractal **+8.1.** So
> "SpatialVLA is weak to pruning" is too coarse; **how much you delete and
> which cell must be said together.**

> ⚠️ **And do not present this contrast as "only the backbone differs."**
> Check what was actually deleted and **the region differs too** —
> SpatialVLA/Bridge's prune 1 deleted **L9·L10·L17** out of 26 layers (varies
> by task; 35–65% in depth, **all middle**), while OpenVLA and UniVLA, whose
> `--depth-min-layer 0.5` restricts candidates to the back half, deleted L16
> or later. And we measured directly that region alone can split the outcome —
> **on UniVLA, holding the count at 4 and only widening the window
> (`--depth-min-layer` 0.5 → 0.08) moved Δ from −2.1 to −79.2** (§4.4). In
> success rates, 81.2% → 2.1%. In the narrow window, **all four tasks chose
> layers L20 or higher** (e.g., `[21,24,26,30]`); widen the window and **all
> four tasks pulled in L2 and L4** (e.g., `[2,4,26,30]`). Deleting the last 8
> of 32 layers costs −4.2 (p = 0.42) — practically free — while two early
> layers take the policy down.
>
> Do not write the deleted layers as **one set** — the ranking is recomputed
> per task from that task's first frame, so the four tasks' sets differ
> slightly. The bracketed sets above are one representative task each; **the
> conclusion rests on the region, not on individual layers.**
>
> In short: **the effect of depth pruning is governed less by "how many
> layers" than by "which layers," and backbone comparisons do not hold until
> the candidate windows are aligned.** This too is something prior work does
> not address — they usually report only the count — and it is the same story
> as the §6 counterexample, which turned out to come from a mis-set candidate
> window.

## 2.5 What already sits in these papers' tables, undiscussed

This is what we learned by reading.

> **First, how this relates to §6.** What this section shows is a fact about
> the published tables: **inside them, `move_near` and `pick_coke_can` move in
> opposite directions, and nobody says so.** That is **a claim about reporting
> practice**, and it stands on its own. What we attempted in §6 was to explain
> the *cause* of that split as "the ability to understand the instruction
> degrades first" — and **our own measurement did not support that
> explanation.** The signal came out the other way — counting failure kinds
> inside one task, what grew was not "moved the wrong object" (1 → 4,
> p = 0.375) but **"touched nothing"** (0 → 12, p = 0.0005) (§6.4). That is,
> **what degrades first is not understanding but doing.**
>
> So do not read this section as "the evidence for §6." This section
> establishes **that the split exists**; §6 establishes **what breaks first in
> one cell.** Do not chain them into "so the splits in others' tables have the
> same cause" — we measured only one cell.

Split SIMPLER's Google Robot tasks apart and **the same pattern repeats in the
tables the prior papers themselves published** (with the two exceptions noted
in ⚠️ below):

- **`pick coke can` (one object, nothing to choose) goes up or holds under
  the interventions.**
- **`move near` (three objects, must pick the named one) goes down.**
- And **where a capacity ladder exists, the direction is monotone.**

> **The third bullet's scope, narrowed.** Of the 12 settings, only
> **EfficientVLA's four configurations** vary capacity in steps (L=28/22 ×
> T=112/56), and they form ladders in two settings each. All four ladders are
> monotone — Visual Matching: PickCan +4.0 → +3.4 → +2.7 → +2.0, MoveNear
> −1.7 → −2.6 → −2.9 → −3.7; Variant Aggregation: +5.2 → +4.8 → +4.3 → +3.6
> and −3.2 → −3.6 → −4.4 → −5.0. The other four rows (FastV 2, VLA-Cache 2)
> are **single points with no capacity axis**, so monotonicity cannot even be
> asked. **Direction: 12/12. Monotonicity: confirmed in 8/12 only.**

Across EfficientVLA, VLA-Cache, and FastV, **all 12 configurations** behave
this way (Visual Matching 6 + Variant Aggregation 6). Yet no paper mentions
it. **They all report only the 4-task average.**

**Sources.** All 12 rows are in EfficientVLA's Table 2. The 2 VLA-Cache rows
are not EfficientVLA's reproduction but **identical to the decimal with
VLA-Cache's own Table 3** (Matching 92.0 / 83.3 / 70.5 / 51.6, Aggregation
91.7 / 79.3 / 32.5 / 45.8) — so they are a citation, and therefore **an
independent source.** The 2 FastV rows exist **only in EfficientVLA** — not an
independent source but their own comparison runs.

The precise statement is **"two author groups, three method families, 12
settings."** Do not write "three papers independently agree."

> ⚠️ **Two things we must disclose first ourselves.**
>
> **① We excluded the 2 Random Dropping rows.** Table 2 has **14 rows**
> besides the baseline, and Random Dropping breaks the pattern — PickCan
> drops 91.3 → **9.7**. Random keeping is not a proposed method but a control
> designed to show that *"what you keep dominates,"* so excluding it is
> defensible — but **we have to say we excluded it.**
>
> **② The `Drawer` family is not clean.** It drops 6/6 in Visual Matching but
> mixes in Variant Aggregation, and `DrawerApple` mostly goes up. **The only
> two tasks that split monotonically are `PickCan` (up) and `MoveNear`
> (down)** — the claim has to be narrowed to those two.

**One level up, there is one more instance of the same structure.** ShortGPT
— the original paper for our depth score — reports that with 25% of layers
removed, **multiple-choice tasks hold or improve (BoolQ 71.62 → 74.71) while
generative tasks fall to near zero** (XSum 19.40 → 0.67), and its Limitation
section states plainly: *"The reasons behind it still need to be explored."*
(Appendix A.3.) The domain differs (LLM text vs. robot control), so **we do
not add it to the 12 settings.** But it shows, one domain up, the same shape:
*cut capacity and abilities do not shrink evenly — some are damaged first and
some even improve.* §6 measured the next step (what dies first) in one cell of
our grid, but that is **a different domain and a different intervention**, so
we do not use this as supporting evidence for §6. The authors' own guess —
**error accumulation** — is worth noting: robot control is a structure where
**your own actions produce the next frame**, so errors compound.

**Neither EfficientVLA nor VLA-Cache mentions this split — we checked the
running text.** VLA-Cache's SIMPLER discussion is, in full, *"success rates
comparable to the CogACT baseline ... while substantially reducing
computational overhead"* — nothing about MoveNear going down. In
EfficientVLA, the word `MoveNear` appears **only inside tables.**

EfficientVLA averages the two opposite-direction curves in its own table and
writes *"merely a 0.6% drop."* **There are two curves going opposite ways
inside their own table, and the text does not discuss them.** Whether the
authors did not see it or saw it and did not write it, we cannot know; what we
can verify is that **it is not in the text.**

**Why this supports us rather than scooping us.** Our claim is "this field's
reporting practice hides a real effect." Evidence for such a claim can only
come **from inside published papers.** There are only three possibilities.

| If | Then |
|---|---|
| the pattern is **absent** from prior tables | our claim has no basis |
| a prior paper **discussed** the pattern | our contribution disappears |
| **it is in the tables and nobody discussed it** | ← **our claim is supported. This is the case.** |

So these papers are not competitors — they are **exhibits.** And the fact
that the pattern shows up outside our own lab pre-empts the objection "maybe
your setup is just odd."

**The only thing we take from others' tables is that one confirmation.**
Everything else we do ourselves — we **pair** episodes (prior work does not
release per-episode records, so it cannot be done to them; that is exactly one
of our points), we **test**, we **cross** backbone and benchmark, and we
**measure what breaks first inside a task** (§6 — where the result came out
against our own hypothesis, which is why we cannot claim to have confirmed
the observation we borrowed from their tables).

## 2.6 The evaluation-methodology genre, and our place in it

The ancestor of the genre this work belongs to is Bag of Tricks for Image
Classification with CNNs (He et al., CVPR 2019); more recently, there is work
re-measuring inference-time methods the same way (NeurIPS'25 D&B). The methods
that paper uses — Best-of-N, beam search, MCTS, self-consistency, self-refine
— were **none of them invented by its authors.** Nobody calls it a copycat
paper.

**The only kind of paper that would directly overlap with our contribution is
one that compares the same intervention across several backbones × several
benchmarks with paired episodes.** What we found:

| Paper | Backbones | Benchmarks | Crossed the axes? |
|---|---|---|---|
| EfficientVLA | CogACT (3 sizes) | SIMPLER only | ✗ |
| VLA-Cache | OpenVLA & OpenVLA-OFT on LIBERO; CogACT on SIMPLER | LIBERO + SIMPLER + real robot | ✗ **switching benchmark also switches backbone, so sign comparison is impossible** |
| MoLe-VLA | CogACT / OpenVLA | RLBench only (+ real robot) | ✗ **two backbones on one benchmark** |
| Gaze-Reg | **Pi-0** (main) · OpenVLA (transfer check) | Pi-0: LIBERO 4 suites + Gym-Aloha + real robot; **OpenVLA: LIBERO only** | ✗ **has both axes, but the crossing cell is empty** — below |
| **This work** | **3** | **2** | **✓ five cells of a 3 × 2 grid** |

**Zero competitors in the range we checked.** But **Gaze-Reg is the closest,
and its shape matches our grid.**

| Gaze-Reg | LIBERO | Gym-Aloha |
|---|:--:|:--:|
| Pi-0 | ✓ | ✓ |
| OpenVLA | ✓ | **✗** |

That is: they have **a backbone axis** (Pi-0 vs. OpenVLA on LIBERO) and **a
benchmark axis** (Pi-0 on LIBERO and Gym-Aloha), but **the fourth cell is
empty, so the two axes never meet.** It is exactly the shape of our empty
UniVLA/Fractal cell — so neither they nor we can ask "does the sign hold on
both axes at once" in a single cell. **The difference is that they did not
attempt the comparison, and for us it is the whole point.** MoLe-VLA also has
a backbone axis (CogAct and OpenVLA on RLBench). VLA-Cache has two benchmarks,
but **the backbone changes with the benchmark**, so a sign flip cannot be
observed in that design even in principle.

**And none of them tests with paired episodes.** Gaze-Reg averages over three
seeds, which handles run-to-run variance (their Table 2) — but that is a
different question from **counting what the intervention broke and what it
fixed from the same initial state.**

> ⚠️ **This is a negative claim.** "None exists" cannot be proven; the table
> above means we did not find one among the five VLA papers we read closely
> and the range we searched. It needs **at least one systematic re-check
> before submission**, and in the paper it should be phrased as "we are not
> aware of."

So the citations above are not losses — they are **a list of citation
obligations.** Four things we cannot claim to have discovered independently:
VLAs tolerate layer deletion (EfficientVLA·MoLe-VLA); pruning sometimes raises
success (EfficientVLA); vision reduction that **only blurs pixels** hurts VLA
(Gaze-Reg, 10/10); vision reduction that **does not cut token count** does
not buy speed (VLA-Cache). **Citing all four and presenting our measurements
as an independent replication is the honest move — and the replication has
value of its own.**

### A closing paragraph for this section (draft)

> Efficiency interventions for VLA policies are evaluated one benchmark and
> one backbone at a time, and reported as a single average success rate. We
> re-measure three of them — time, vision, and depth — across two benchmarks
> and three backbones (UniVLA on Bridge only; its public checkpoint is
> Bridge-only), with per-episode pairing. We find **(i)** the sign of the
> effect is not stable along either axis; **(ii)** the average hides a
> consistent split — between tasks that require picking out the named object
> and tasks that do not — **which already exists, unmentioned, in the
> published tables of the three prior methods**; and **(iii)** on the vision
> axis, the compute that the intervention is said to save measures zero.

**Sources.** The raw reading log is `RelatedWork_Reading.md`; the full list of
60 candidate papers with priorities is `RelatedWork_Candidates.md`.

---
---

# Appendix A. Per-paper notes — seven questions

> If §2 is the **per-axis synthesis**, this appendix is its **raw material.**
> Each paper is taken apart with the same seven questions, and the answers
> feed §2's (d) and (e). The question set is the same one used by the lab's
> 360-segmentation work log.
>
> **Questions 6 and 7 are where our contribution sits.** If "were the
> comparisons and ablations enough to support the idea" can be answered *no*,
> and the reason why is the thing we measure, that is our paper's opening.
>
> At the head of each entry we say how far we checked — what we saw directly
> in the original's tables and text, versus what rests on earlier notes.

---

## A.1 EfficientVLA: Training-Free Acceleration and Compression for VLA

`arXiv 2506.10100v1` · original checked (Table 2, 8 pages of text)

**1. What problem does it address?** Diffusion-based VLA inference is slow.
The authors split the slowness into three places — the language model's depth,
visual tokens that add nothing, and the action-refinement process repeating
computations whose successive steps are nearly identical. The goal is to cut
all three at once, **without training.**

**2. How is the problem framed? Is the direction new?** The problem is not
new. Their framing is that earlier attempts touched **one module at a time**,
which only moves the bottleneck — *"optimizing one module in isolation merely
shifts bottlenecks."* Figure 1(a) shows visual-token pruning helping at first,
then *"efficacy quickly diminishes as the system becomes memory-bound by the
LLM."* **The shape of the framing resembles ours** — look at one axis only and
you conclude wrongly.

**3. Core idea and its assumption.** From the observation that hidden states
of adjacent layers have high cosine similarity (Figure 1b), they conclude
**layers are redundant**; from action-refinement steps having similar
features, that **the repetition is redundant too.** The assumption: "units of
computation that resemble each other can be skipped without changing the
output much." **The same assumption as our depth pruning.**

**4. Technical contributions.** ① layer pruning (L=28 or 22); ② visual token
selection that weighs task relevance and diversity together (T=112 or 56);
③ static caching that refreshes the diffusion head's attention/MLP features
every N steps.

**5. What comparisons did they run?** SIMPLER environment, CogACT backbone,
two settings (Visual Matching, Variant Aggregation), four tasks (PickCan /
MoveNear / Drawer / DrawerApple). Baselines: Random Dropping, FastV,
VLA-Cache. Headline result: **FLOPs down to 28.9% (a 71.1% cut), 1.93×
speed-up, average success −0.6 points.**

**6. Were the ablations enough? — this is where we look.**
The ablations themselves are diligent. They show Random Dropping falling
74.8 → **20.9**, establishing that "what you keep dominates," and they sweep
layer count and token count on a grid.

But **the unit of reporting is the 4-task average.** Unfold that average and
in **all 12 settings** of Table 2, `PickCan` rises while `MoveNear` falls.
Just within EfficientVLA's own four configurations, ordered by capacity, the
split widens monotonically: `+4.0 → +3.4 → +2.7 → +2.0` versus
`−1.7 → −2.6 → −2.9 → −3.7`. The authors summarize the setting as *"merely a
0.6% drop in average success rate."*

> **So there are two opposite-direction curves inside their own table, and
> the average hides them.** All we can verify is that the text does not
> discuss it — whether the authors missed it or chose not to write it, we
> cannot know. This is the basis of our §2.5, and the reason our protocol
> (episode pairing + task-family breakdown) is needed. Scope notes: the
> `Drawer` family's direction is not clean, and the 2 Random Dropping rows
> are a deliberately damaging control, so we excluded them — both facts we
> disclose ourselves.

**7. Open questions it leaves.** ① One backbone (CogACT) — whether the same
score gives the same sign in another policy family is untested. ② No
per-episode records released, so a third party cannot re-test with pairing.
③ No mention of `MoveNear`'s consistent decline.

---

## A.2 VLA-Cache: Efficient VLA Manipulation via Adaptive Token Caching

`Univ. of Sydney / SJTU` · original checked (Table 2, Table 3, §5.3)

**1. What problem does it address?** In manipulation, **consecutive frames
are nearly identical**, and recomputing everything every step wastes that.

**2. How is the problem framed?** It starts from the point that VLM
acceleration methods (FastV, SparseVLM, ToMe, …) *"reduce redundancy within a
single image but disregard the temporal and spatial structure essential for
robotic tasks under closed-loop control."* That is, **do not carry a
single-image technique into closed-loop robotics unchanged** — which rhymes
with our "do not carry a number measured in one condition into another."

**3. Core idea and assumption.** Find visual tokens that did not change
between frames and reuse their KV entries. Assumption: "a static token's
representation stays valid across steps."

**4. Technical contributions.** ① identifying static tokens between frames;
② a task-relevance filter; ③ a layer-adaptive strategy that varies the reuse
ratio by each layer's attention concentration.

**5. What comparisons did they run?** LIBERO's four suites (OpenVLA, and once
more with OpenVLA-OFT), four SIMPLER tasks (CogACT), and four real-robot
tasks on a Kinova Jaco2. All on an RTX 4090. **This is where the key table of
our §2.3 (d) comes from:**

| | FLOPs (T) | Latency (ms) |
|---|---|---|
| OpenVLA | 1.864 | 51.91 |
| + FastV | **1.864** (no change) | **53.28** (up) |
| + SparseVLM | 1.407 (−24.5%) | **83.39** (+60.6%) |
| + VLA-Cache | 1.355 | 31.83 |

The two causes the authors name: these methods *"target long output
sequences, whereas VLA models generate short action outputs (e.g., 7
tokens)"*, and they operate within a single frame, which can *"disrupt
spatial fidelity."*

**6. Were the ablations enough?** The method's own ablation (static tokens
only / + task filter / + layer-adaptive) is diligent. The problem is that
**the backbone changes with the benchmark** — OpenVLA on LIBERO, CogACT on
SIMPLER. So **how the effect changes when the benchmark changes cannot be
observed in this design even in principle**: backbone and benchmark move
together. We separate those axes.

Also, the per-task numbers in their SIMPLER Table 3 match EfficientVLA's
VLA-Cache row to the decimal — confirming the two are independent sources.
And here too, `PickCan` rises (91.3 → 92.0) while `MoveNear` falls
(85.0 → 83.3). The paper's SIMPLER discussion is, in full, *"success rates
comparable to the CogACT baseline ... while substantially reducing
computational overhead"* — **no mention of the split.**

**7. Open questions it leaves.** ① Benchmark and backbone are **entangled and
cannot be separated**; ② no per-episode records; ③ it publishes the
important measurement that FLOPs and latency diverge, but never tests whether
that applies to **vision-axis techniques in general.**

---

## A.3 ShortGPT: Layers in LLMs are More Redundant Than You Expect

`arXiv 2403.03853v3 (2024-10-11)` · original checked (Table 2, §5 Limitation)

**1. What problem does it address?** If many of an LLM's layers mostly pass
their input through unchanged, can we find and remove them to shrink the
model? The starting observation (their Figure 2) is that in pre-norm
architectures, a layer's input and output become unusually similar.

**2. How is the problem framed?** Earlier pruning worked mostly along the
width dimension; the claim is that **slack in the depth direction** has been
underrated. Not a new problem, but "simple layer deletion beats complicated
schemes" is new as a direction.

**3. Core idea and its assumption.** **Block Influence**:

```
BI_i = 1 − E[ cos( X_i , X_{i+1} ) ]
```

The procedure: pass **unlabeled text such as PG19** through the model once
(this input is the calibration data), collect each layer's input and output,
compute BI, and **delete from the lowest BI up.** Low BI means input and
output are nearly the same — **a layer that did nothing** — so the assumption
is it can go. **This is the score our depth pruning uses.**

**4. Technical contributions.** The BI metric; training-free layer deletion
based on it; and showing it does not interfere with quantization (they
compose). **Their §4.4** extends it to non-transformers (Mamba, RWKV).

**5. What comparisons did they run?** Llama2-7B/13B, Baichuan2-7B/13B, 13
benchmarks (CMNLI, HellaSwag, PIQA, CHID, WSC, CoQA, BoolQ, Race-H/M, XSum,
C3, MMLU, CMMLU). Baselines: LLMPruner, SliceGPT, LaCo. Average retention
beats the competitors (86.31% vs. LaCo's 80.39% on Llama2-7B).

> ⚠️ **The paper disagrees with itself — we have to choose which number to
> cite.** The introduction (p. 2) gives as its headline: *"removing 10 layers
> (25% of the total 40 layers) from the LLaMA 2-13B model resulted in only a
> slight drop ... on the MMLU benchmark, **from 55.0 to 52.2**."* But **Table
> 2's same setting** (Llama2-13B, ShortGPT, ratio 24.6%) lists MMLU as
> **55.00 → 54.69.** A 2.5-point discrepancy.
>
> There is one more of the same kind — Llama2-7B ShortGPT's BoolQ is **74.71
> in Table 2 and 74.41 in Table 6** (averages 41.24 vs. 41.22). Small, but it
> means the paper's tables do not fully agree with each other.
>
> We cannot judge which is right. So **we use the table value (54.69) and
> note that the introduction differs** — the rest of our numbers in this
> appendix come from Table 2, and it is better not to mix sources.

**6. Were the ablations enough? — here the same structure as our §6 shows
up.**

Unfold Table 2's Llama2-7B row by benchmark, at 27.1% removal:

| Benchmark | Type | Dense | ShortGPT | Δ |
|---|---|---|---|---|
| BoolQ | multiple choice | 71.62 | **74.71** | **+3.09** |
| Race-M | multiple choice | 34.19 | **35.17** | **+0.98** |
| MMLU | multiple choice | 45.39 | 43.96 | −1.43 |
| CMMLU | multiple choice | 32.92 | 32.25 | −0.67 |
| C3 | generative | 43.56 | 39.62 | −3.94 |
| **XSum** | **generative** | **19.40** | **0.67** | **−18.73 (near zero)** |
| Average | | 47.78 | 41.24 | −6.54 |

**Multiple-choice tasks hold or improve; generative tasks fall to near
zero.** And the authors put this in their Limitation section themselves:

> *"the negative effect of layer removal is more significant on generative
> tasks compared to multiple-choice tasks. When we remove 25% layers from
> Llama2-7B or Baichuan2-7B, the performance in generative tasks such as XSum
> and C3 deceases to nearly zero ... **The reasons behind it still need to be
> explored.**"*

One more: **the drop is milder at 13B** (XSum 23.45 → 17.59). So the split
**also depends on model size.** The authors' guess is that generative tasks
suffer **error accumulation**, and that bigger models resist it better.

> ⚠️ **But C3 should be read out of that quote.** The authors write *"XSum
> and C3 deceases to nearly zero,"* yet in the same paper's Table 2, **C3
> does not go near zero** — Llama2-7B 43.56 → **39.62**, Baichuan2-7B
> 64.55 → **56.33.** The only benchmark that falls to near zero is **XSum**
> (0.67, 0.04). The direction (generative tasks are hurt more than multiple
> choice) matches the table; **"nearly zero" is true of XSum only.** This is
> the third internal inconsistency we found in this paper.

> **What this means for us.** *Cut capacity and abilities do not shrink
> evenly — some are damaged first, and some improve* — this shape **already
> exists inside the original paper for our score, and the authors state they
> cannot explain it.**
>
> But the domain differs (LLM text vs. robot control), and **what we measured
> in §6 is a different intervention in one robot cell.** So we do not use
> this as supporting evidence for §6, and we do not add it to §2.5's 12
> settings. We cite it only as the same shape appearing one domain up.
>
> Their guess — **error accumulation** — is worth attention: robot control is
> a structure where your own actions produce the next frame, so if the guess
> is right, our side should be damaged *more*. **We did not test that
> prediction.**

**7. Three things the paper leaves unanswered.**

① **Why the split depends on the task.** Delete 25% of layers and
multiple-choice scores *rise* (BoolQ 71.62 → 74.71) while text generation
falls to near zero (XSum 19.40 → 0.67). The authors found the split but write
that they do not know the reason — *"The reasons behind it still need to be
explored."*

② **Never tried on a robot.** All experiments are text tasks (13
benchmarks). Whether "deleting layers is fine" carries to robot control
cannot be known from this paper.

③ **Filling the gap and retraining recovers a little — why is not
addressed.** Their §4.6 puts **a small MLP where the deleted layers were and
retrains.** XSum climbs 0.67 → **4.89**, but the original was 19.40 — **far
short of a recovery** — and the average (41.22 → 43.16) stays below Dense
(47.78).

> **What these three mean for us.** ① has the same shape as our §6 — cut
> capacity and abilities do not fall evenly; **something specific breaks
> first.** ② is why our work is needed. ③ is a remedy our training-free
> setting cannot use at all.

> **Two differences from us.** To choose layers you must pass something
> through the model. ShortGPT feeds *"a calibration set, which is a set of
> unlabelled text samples such as PG19,"* and averages BI **across those
> samples** (the `E[ ]` in the formula). We feed **one frame — the first
> observation of the run** — and average only across token positions within
> that single forward pass.
>
> **① What is looked at** — task-unrelated text vs. the task's own first
> frame. **② How much is looked at** — many passages vs. one frame.
>
> One thing that is the same: **neither side re-measures during the run.**
> Our code calibrates once per run behind a `calibrated` flag
> (`tome_spatialvla_eval.py:299`, `adaptive_sparse_vla/depth_prune.py:246`)
> and the layer set is fixed afterward. Layers differ per task in the result
> files because **each task is a separate process** calibrating on its own
> first frame (`[2,4,6,23]` on `move_near`, `[2,4,23,26]` on pick).
> Per-episode recalibration does not exist in our implementation.
>
> **Whether these two differences help or hurt, we did not measure.** What
> §4.4 (c) shows is that **selection strongly shapes the outcome**, so
> changing the calibration data could change the selection. And since **the
> single-frame side is us**, whether a ranking chosen from the first frame
> represents the whole episode is **our open question, not theirs.**

---

## A.4 MoLe-VLA: Dynamic Layer-skipping VLA via Mixture-of-Layers

`arXiv 2503.20384v2 (2025-04-14)` · original checked (Table 1, Table 5)

**1. What problem does it address?** Run a VLA on half its LLM layers while
keeping performance. Their criticism of prior sparsification (early exit,
token pruning) is that it *"neglect[s] the critical role of the final layers
that encode the semantic information most relevant to downstream robotic
tasks."*

**2. How is the problem framed?** It starts from the homogeneity of LLM
layers but adds the observation that **some layers matter specifically for
robot tasks.** That is, the problem is "which layers you skip."

**3. Core idea and assumption.** A small selector that picks which layers to
run per input (the **STAR router**, Spatial-Temporal Aware Router), plus
**CogKD** (self-knowledge distillation, EMA teacher α = 0.999) to restore the
cognition lost by skipping. Objective:
`L = L_task + λ₂·L_cog + λ₃·L_lb` (λ₂ = 0.5, λ₃ = 0.1).

**4. Technical contributions.** Turning different layers on per input; the
selector; and distilling the full-depth model's judgment into the reduced one
(self-distillation). The combination of the three.

**5. What comparisons did they run?** **RLBench, 10 tasks + a real robot.**
No LIBERO, no SIMPLER. All five efficiency baselines use **50% of the LLM
layers.**

| Method | Mean Acc. | FLOPs (G) |
|---|---|---|
| OpenVLA | 45.4% | 1930.0 |
| CogAct | 57.2% | 1935.8 |
| **Random-skip-CogAct** | **51.2% (−6.0)** | 984.3 |
| MoD-CogAct | 56.4% (−0.8) | 985.8 |
| DeeR-CogAct | 59.2% (+2.0) | 997.4 |
| **MoLe-OpenVLA (theirs)** | **55.6% (+10.2)** | 981.5 |
| **MoLe-CogAct (theirs)** | **60.8% (+3.6)** | 985.8 |

**6. Were the ablations enough? — this is where the source of the gain
splits.**

Table 5's component ablation:

| | STAR | Cognition | CogKD | Mean |
|---|---|---|---|---|
| Ex0 (=CogAct) | ✗ | ✗ | ✗ | 57.2% |
| Ex1−1 | ✓ | ✗ | ✗ | **56.3%** |
| Ex2−1 | ✓ | ✓ | MSE | 58.3% |
| Ex2−3 | ✓ | ✓ | Reserve KL | 59.4% |
| **Ex2−4 (final)** | ✓ | ✓ | MSE + Reserve KL | **60.8%** |

**The router alone lands at 56.3% — below the baseline (−0.9).** The gain
arrives **after distillation is added.** So what this table supports is less
"the layer selector picks layers well" and more **"skipping layers costs
performance, and only after distillation restores it does a gain appear."**

That said, random skipping costs −6.0 in the same table. So two sentences
hold together: **you cannot skip arbitrary layers, and even good picks gain
nothing without distillation.**

Conditions to attach when citing: **training is required** (ours is
training-free); **one simulation benchmark, RLBench** (plus a Franka FR3 real
robot, 3 tasks); and, as the paper states, **25 trials per task** with no
confidence intervals and no paired testing.

> ⚠️ One ambiguity in the table. The `Random-skip-CogAct` row lists its
> action head as **MLP**, but `CogAct`'s head is Diffusion. By its name it
> should be Diffusion, so either the entry is a typo or a genuinely different
> head was used — the paper does not say. **Check this before citing −6.0 as
> a clean control for CogAct.**

**7. Open questions it leaves.** ① Does the conclusion hold for a
training-free, fixed schedule — which is the setting we measure? ② Does it
transfer outside RLBench? ③ Per-task success scatters widely (`Sweep to
Dustpan` 4.0%–72.0%) and the paper does not discuss that variance.

---

## A.5 Gaze-Regularized VLA Models for Robotic Manipulation

`arXiv 2603.23202` · Pani & Yang, HKU · CVPRW 2026 (GRAIL-V) · original
checked (Appendix D.2, Table 11)

**1. What problem does it address?** VLAs have **no mechanism for actively
allocating visual attention.** Human gaze carries intention, planning, and
execution, so use it as a supervision signal — that is the body of the paper.

**2. How is the problem framed?** From the observation that in fine
manipulation the model does not know where to look. Existing approaches
(dynamic zoom, crops, foveated imagery) demand architecture changes or
mappings — that is their criticism.

**3. Core idea and assumption.** Turn the gaze heatmap into a patch-level
distribution and **regularize the transformer's attention with a KL
divergence.** No architecture change, no inference-time cost.

**4. Technical contribution.** The gaze-regularized training framework. The
abstract's summary: **4–12% improvement** on manipulation benchmarks, equal
performance in fewer training steps, robustness to lighting changes and
sensor noise.

**5. What comparisons did they run?** **The main backbone is Pi-0** — LIBERO's
four suites (Spatial / Object / Goal / 10) and two Gym-Aloha tasks run on
Pi-0 (Table 2, three-seed averages), and the three real-robot tasks are Pi-0
as well (Table 4). **OpenVLA is a transfer check** — does it work on a
different architecture — and runs LIBERO only (Table 3, 68.5 → 74.2). **What
matters to us is not the body but Appendix D.2** — a variant that builds
foveated RGB around the gaze peak and feeds it to the standard encoder,
trained for 30k steps. Table 11: **all 10 tasks drop, overall 85.9 → 78.5
(−7.4).** The authors' reading: *"aggressively reducing peripheral detail
removes useful contextual cues (e.g., table geometry, supporting surfaces, or
alternative grasps) that the policy relies on for precise spatial
reasoning."*

**6. Were the ablations enough?** The appendix's foveation experiment is **a
single point (one strength, one centering rule).** With no dose-response
curve, the drop cannot be separated into "foveation itself" versus "this
strength." This paper reports 10/10 down at one point; Look, Focus, Act
reports gains under different conditions — so this is **not an axis you can
settle from one point.** That is why we planned the keep-percent sweep, and
indeed moving the strength took us from +4.2 to +30.2 (§4.3 b).

> **Two conditions to attach when citing.** ① Their foveation is applied
> **during training** (the section title is "Foveated Vision *during
> Training*"). We apply it at inference, without training. ② Their center is
> the **peak of human gaze**; ours is fixed at the image center. Both
> conditions **favor foveation** — they trained on that input and aligned the
> center with human gaze, and still dropped 10 of 10. Our setting is less
> favorable on both counts and went *up* — so our result is not explained by
> this paper.

**7. Open questions it leaves.** The dose-response over strength, and the
behavior when applied **only at inference, without training.** Both are what
we measure.

---

## A.6 Look, Focus, Act (+ Segment This Thing)

`arXiv 2507.15833v2 (2025-09-22)` · Chuang et al., UC Berkeley / Tongji / UC
Davis · original checked (Table I, Table II, §5 A)
`arXiv 2506.11131v1 (2025-06-10)` · Schmidt & Newcombe, Meta Reality Labs ·
CVPR 2025 · original checked (Table 2, §5)

The two are grouped because **the same tokenization crosses two settings.**
As Look, Focus, Act itself states, the foveated tokenization comes from
Segment This Thing — *"we adopt the foveated patch tokenization method
introduced in [18] for image segmentation and adapt it for robot learning."*

**1. What problem do they address?** Both: **the vision encoder is too
expensive.** Both take the same way out — instead of shrinking the model,
**cut patches non-uniformly to reduce the token count.** Small dense patches
at the center, larger and sparser ones outward, all scaled to one size and
fed to an ordinary ViT.

**2. How is the problem framed?** Segment This Thing points out that prior
SAM-family slimming **all shrank the model**, and shrinks the input instead.
Look, Focus, Act points out that robot policies **treat the image
uniformly**, and foveates around a human gaze point.

**3. Core idea and assumption.** The assumption: "the task-relevant part of
the image is small; the rest can be coarse." **The same hypothesis as our
§2.3 (c).**

**4. Technical contributions.** STT: variable-resolution patch tokenization
and a mask decoder matched to it. Look, Focus, Act: carries it to robot
policies and compares two ways of **predicting** gaze (a separate model
first, or jointly inside the action space).

**5. What comparisons did they run? — this is the part that matters to us.**

STT (single prompt, one image, RTX 3080):

| Model | Latency (ms) | GFLOPs |
|---|---:|---:|
| SAM-H | 572.7 | 6533.7 |
| EfficientSAM-S | 78.6 | 489.4 |
| MobileSAM | 20.7 | 124.4 |
| **STT-L** | **13.7** | **108.0** |

Cutting tokens to about 1/24 of SAM's makes it **about 42× faster than
SAM-H** and about 5.7× faster than EfficientSAM-S (both computed from the
latencies in the table — 572.7/13.7 and 78.6/13.7). **Nothing repeats after
the encoder, so what the encoder saves becomes the system's gain, one for
one.**

Look, Focus, Act (batch 64):

| | ViT tokens | ViT latency (ms) | ViT GFLOPs |
|---|---:|---:|---:|
| Fine (uniform 18×18) | 324 | 243.8 | 1905.4 |
| **Foveated** | **20** | **16.4** | **115.6** |

The ViT alone: **14.9×**, GFLOPs **−93.9%.** But whole-policy inference goes
334.7 → 87.9 ms — **3.8×** (training: 833.2 → 108.2 = 7.7×).

**6. Were the ablations enough? — and what the authors say themselves.** The
reason for the gap is in their own text:

> *"During inference, differences are smaller because the flow matching
> transformer runs multiple sampling steps (8 in our case) while image
> features are processed only once by the ViT."*

**The vision stage runs once; the later stages run many times.** Just 8 runs
of a flow transformer turned 14.9× into 3.8×. **In a VLA the later stage is
the whole decoder, 12–26 times** — so the same logic bites much harder. That
computation is the last two tables of our §2.3 (d).

On the success side, the picture depends on conditions (their Table III, sim;
Table IV, real robot), **so it must be reported split:**

**① No pre-training, no distractors (sim).** Fov-UNet matches or beats the
other three on **all six tasks** (e.g., HookPackage 28/18/12 → **56**,
PourTestTube 34/60/78 → **84**). Fov-Act also beats Fine and Coarse on
`ThreadNeedle` (62 vs. 57·48) and `PourTestTube` (78 vs. 34·60).

**② Add distractors and foveation's edge grows** — the authors read this as
trimming the periphery buying **robustness to distractors.**

**③ But MAE pre-training shrinks the gap.** Without distractors, **Fine wins
two tasks and Fov-UNet three.** So "foveation wins" is **clearest when there
is no pre-training.**

**④ On the real robot, Fine is ahead.** Of Table IV's four cells, **Fine
takes three** — Ball 64 vs. 62 (standard), Toothbrush 24 vs. 18 (standard)
and 18 vs. 14 (distractor). Fov-UNet leads only in the Ball-distractor cell
(48 → **56**). The authors themselves write *"differences ... are less
pronounced than in simulation."*

> **The single most useful line for us is `HookPackage` under Fov-Act.** On
> that task alone, Fov-Act is worse than every other method (12). The authors
> give the reason: **the target — a small hook — sits in the periphery, and
> foveated downscaling makes the gaze miss it.** That is, **trimming the
> periphery is a direct loss whenever the target is in the periphery.** It
> points the same way as Gaze-Reg's "the periphery is needed context," and it
> also matches the shape of our grid, where foveation's sign flips from cell
> to cell — **there are cells that need the periphery and cells where it gets
> in the way.**

> ⚠️ **Three conditions to attach when citing.** ① The method changes the
> tokenization pattern itself, so **public pre-trained ViT weights cannot be
> used.** The authors ran MAE pre-training themselves, on a **60k-image
> subset** of ImageNet-1K — and state that the Fine/Coarse numbers could rise
> with weights like DINOv2. ② **It needs human gaze** (VR headset eye
> tracking). ③ The latency numbers are at **batch 64**, not the batch-1 of
> robot execution.

**7. Open questions they leave.** ① **How much of the gain survives on a VLA
with an autoregressive decoder** — neither paper measures that setting; our
profile fills that slot. ② Does a success gain appear with a **fixed center
and no gaze**? Our OpenVLA/Bridge +18.8 is that condition — but they train
and we do not. ③ Whether the pixel-only variants (ours, Gaze-Reg's appendix)
and the token-cutting variants (these two papers) **affect success
differently** — nobody has measured them side by side.

---

## A.7 The seven papers in one line each

| Paper | Backbones | Benchmarks | Training needed | Episode records released | Axes crossed |
|---|---|---|---|---|---|
| EfficientVLA | CogACT (3 sizes) | SIMPLER only | ✗ | ✗ | ✗ |
| VLA-Cache | OpenVLA·OpenVLA-OFT on LIBERO, CogACT on SIMPLER | LIBERO + SIMPLER + real robot | ✗ | ✗ | ✗ (backbone changes with benchmark) |
| ShortGPT | 4 LLMs | 13 NLP benchmarks | ✗ | — | no robot experiments (but **records the task-type split in its own Limitation**) |
| MoLe-VLA | CogACT / OpenVLA | RLBench + real robot (Franka FR3) | **✓** | ✗ | ✗ |
| Gaze-Reg | **Pi-0** (main) · OpenVLA (transfer) | Pi-0: LIBERO 4 suites + Gym-Aloha + real robot / OpenVLA: LIBERO only | **✓** | ✗ | ✗ (has both axes; the crossing cell is empty — §2.6) |
| Look, Focus, Act | own flow-matching policy | AV-ALOHA sim, 6 tasks + real robot, 2 tasks | **✓** (self-pre-trained ViT via MAE) | ✗ | ✗ |
| Segment This Thing | own ViT (B/L/H) | 9 segmentation datasets | **✓** | — | no robot experiments |
| **This work** | **3** | **2** | **✗** | **✓** | **✓** |

The last two columns are the ones we fill. The qualifier from §2.6 applies
unchanged — **within the range we checked**, we found no prior work that
crossed the axes; that is not a proof that none exists.

> The bottom two rows are **not VLA papers** — they do not compete with us on
> the last column. They are used in §2.3 (d) to show that **the payoff of
> vision reduction depends on what comes after the encoder**, and they are
> not counted in §2.6's "zero competitors."
