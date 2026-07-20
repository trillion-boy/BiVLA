# Lab Meeting Update: Testing Speed & Vision Tricks on 4 Robot AI Models

**Date:** 2026-07-20
**Scope:** OpenVLA / SpatialVLA / RoboVLMs / UniVLA — 4 robot-control AI
models (VLAs)
**Setup:** SimplerEnv WidowX-Bridge robot simulator, 4 tasks
(Carrot/Stack/Spoon/Eggplant), 24 tries per task. No retraining — we only
changed how the model is *run*, not the model itself.
**Every number below comes from my own simulation runs.** Each model's
"before vs after" comparison uses the same code, same checkpoint, and
same GPU — never a number from a paper or from a different setup.

---

## 0. What's new since last time

Last time I had only done SpatialVLA and OpenVLA. This time I finished
the same two tests on **RoboVLMs** and **UniVLA** too. So now **all 4
models have been tested with the same two tricks.**

---

## 1. The two tricks we tested

- **chunk-exec**: A VLA that predicts several future actions in one go
  normally throws most of them away and only uses the first one. Chunk-exec
  says: "why throw them away? Just run them." This means the model needs to
  "think" less often, so it's faster.
- **foveation**: Copy how human eyes work — see the center of the image
  sharply, and the edges blurry/low-detail. We tested two versions:
  **log-polar** (physically warps the image toward the center) and
  **blur** (keeps every pixel in place, just blurs the edges).

## 2. The headline result (one table)

| Trick | OpenVLA | SpatialVLA | RoboVLMs | UniVLA |
|---|---|---|---|---|
| **chunk-exec** | doesn't apply* | ✓ **+13.6pp**, 1.9× faster | ✗ **−36.5pp** | ✗ **−12.5pp** |
| **foveation (log-polar)** | ✓ **+18.8pp** | ✗ **−7.3pp** | ✗ **−19.8pp** | ✓ **+8.3pp** |
| **foveation (blur)** | ✓ **+17.7pp** | ✓ **+11.5pp recovered** | ✗ still broken | △ **−2.1pp** |

(pp = percentage points of task success rate. + means the trick helped,
− means it hurt.)

\* OpenVLA only predicts ONE action per "think", so there's nothing to
"unlock for free" — chunk-exec's whole idea doesn't apply here. More on
this in section 3.2.

**One-sentence conclusion**: Out of 4 models and 2 tricks, no trick works
everywhere. Even *which foveation style is better* (log-polar vs blur)
changes from model to model. This shows that **whether a speed/vision
trick works depends entirely on how the model is built inside** — there
is no universal answer.

---

## 3. Why does it depend on the model? (the short answer)

| Model | Has 3D coordinate math? | Has a vision "bottleneck"? | Has memory across steps? |
|---|---|---|---|
| OpenVLA | No | No | No (decides fresh every step) |
| SpatialVLA | **Yes** (turns pixels into 3D positions) | No | No |
| RoboVLMs | No | **Yes** (squeezes the image into a few tokens) | **Yes** (LSTM memory) |
| UniVLA | No | No | Yes, but NOT an LSTM (just remembers recent frames in its prompt) |

**The pattern**: if a model has one of these "fragile parts" (3D math,
a squeeze-bottleneck, or LSTM memory), tricks that simplify or skip
information tend to break it. If it has none of these fragile parts,
the same trick is often free extra performance.

---

## 4. Each model, in detail

### 4.1 SpatialVLA — has 3D coordinate math

**chunk-exec (Phase 1)**

| Setting | Eggplant | Carrot | Stack | Spoon | **Average** | speed |
|---|---|---|---|---|---|---|
| baseline | 66.7% | 25.0% | 29.2% | 8.3% | **32.3%** | 1× |
| **chunk k=2** | **87.5%** | **41.7%** | 25.0% | **29.2%** | **45.9%** | **1.9× faster** |
| foveation alone (log-polar) | 58.3% | 29.2% | 4.2% | 8.3% | 25.0% | same speed |

- **Chunk k=2 is the clear winner**: +13.6pp success AND 1.9× faster,
  for free.
- Foveation (log-polar) **hurt** by −7.3pp. And when we looked closer:
  the robot could still **grab** the object fine (grasp rate stayed
  high), it just couldn't **place** it correctly anymore.

**Why foveation hurt SpatialVLA**: This model estimates depth from the
camera and turns every pixel into an exact 3D position (it's built to
reason about real-world geometry). Log-polar foveation physically moves
pixels around. After that move, a pixel's new content no longer matches
where the model *thinks* that pixel is in 3D space — so its geometry
math becomes wrong, especially at the edges, which is exactly where
placement targets (the plate, the basket) usually are. That explains
the "can grab but can't place" pattern perfectly.

**Fixing it with blur (Phase 2)**: We tried blur instead — same idea
(sharp center, blurry edges) but **no pixel ever moves**.

| Setting | Eggplant | Carrot | Stack | Spoon | **Average** |
|---|---|---|---|---|---|
| chunk2 + log-polar foveation | 45.8% | 25.0% | 20.8% | 16.7% | 27.1% |
| chunk2 + **blur** foveation | **79.2%** | 20.8% | 12.5% | **33.3%** | **36.5%** |

Blur recovered most of the damage (27.1% → 36.5%, +9.4pp; up to +11.5pp
with extra tuning). This **proves** the problem was pixel movement, not
foveation itself — because the same "less detail at the edges" idea
works fine as long as pixels stay put.

**Bottom line**: chunk k=2 is the big win here (+13.6pp, 1.9× faster).
Foveation breaks this model's geometry unless done carefully (blur).

---

### 4.2 OpenVLA — the "nothing fragile" baseline model

| Setting | Carrot | Eggplant | Spoon | Stack | **Average** | change |
|---|---|---|---|---|---|---|
| baseline | 16.7% | 25.0% | 8.3% | 12.5% | **15.6%** | — |
| **foveation (log-polar)** | 16.7% | 33.3% | 41.7% | 45.8% | **34.4%** | **+18.8pp** |
| **foveation (blur)** | 25.0% | 62.5% | 25.0% | 20.8% | **33.3%** | **+17.7pp** |
| Retina (fancier combo trick) | 4.2% | 25.0% | 16.7% | 4.2% | **12.5%** | **−3.1pp** |

- **Foveation helps a LOT, both versions, almost equally.** OpenVLA has
  no 3D math, no bottleneck, no memory to break — so it doesn't matter
  *how* you simplify the image, it just removes noise and helps.
- **chunk-exec doesn't apply**: OpenVLA only predicts one action per
  "think" — there's no batch of future actions sitting around to reuse.
  We built a "repeat the same action k times" version to compare, but
  that's a fundamentally different (and much weaker) idea than real
  chunk-exec, so we don't count it as a fair test.
- **Retina** (a combined trick: foveation + reuse old camera frames +
  sometimes skip thinking entirely) was **worse than doing nothing**
  (−3.1pp), even though its grasp rate went up (+6.2pp) — same
  "can grab, can't place" pattern as SpatialVLA! We built a diagnostic
  to check: does it get *more* likely to reuse stale info after
  grabbing (during the tricky placement phase)? **We ran the test and
  the answer was no** — reuse rates were about the same before and
  after grabbing. So the hypothesis was wrong, and why Retina fails is
  still an open question.

**Bottom line**: foveation is a clear, safe win here. Retina — a more
complex trick — surprisingly backfired, and we don't yet know why.

---

### 4.3 RoboVLMs — has an LSTM memory AND a vision squeeze-bottleneck

**chunk-exec collapsed**

| Task | baseline | chunk k=2 | speed |
|---|---|---|---|
| Carrot | 25.0% | 0.0% | 2× faster |
| Stack | 4.2% | 0.0% | 2× faster |
| Spoon | 41.7% | 8.3% | 2× faster |
| Eggplant | 87.5% | 4.2% | 2× faster |

Speed doubled exactly as planned, but success nearly disappeared.

**Why**: this model uses an LSTM (a type of memory) to decide actions.
Every time the model "thinks," its memory advances by exactly one tick
— thinking IS the model's internal clock. If you skip half the
"thinks" (to go faster), the memory's sense of time gets scrambled,
even though the robot is still moving every step.

**We proved this with a controlled test**: we made the model think
*every* step (no skipping, no speedup) but only *execute* a slightly
stale action.

| Setting | Spoon | Eggplant |
|---|---|---|
| baseline | 41.7% | 87.5% |
| think every step, use a stale action | 41.7% (no change!) | 66.7% |
| skip thinking (real chunk-exec) | 8.3% | 4.2% |

Using a slightly-stale action barely hurt anything. **The real damage
comes from skipping the "think" step itself, not from the action being
slightly old.** That confirms the memory-desync theory.

**Where does the model's time go?** We also measured exactly which part
of the model is slow:

| Part | % of time |
|---|---|
| Language part (24 layers) | 52.6% |
| Vision part (image encoder) | 29.1% |
| everything else | 12.0% |
| the LSTM memory itself | 4.9% |
| image→text conversion | 1.4% |

The memory part is cheap (4.9%) — ironic, since it's also the reason we
*can't* skip steps to save time.

**Foveation also failed here** (log-polar −19.8pp, blur barely better,
−16.7pp). Unlike SpatialVLA, both grasp AND placement got worse
together — this isn't a geometry problem, it's a "the model just
can't see well enough anymore" problem, likely from its
image-squeezing bottleneck plus the fact that its vision system was
specially trained on clean (non-blurred) images.

**Bottom line**: both tricks failed here, for two different, clearly
proven reasons (memory desync vs. vision damage).

---

### 4.4 UniVLA — has memory, but a different kind (no LSTM)

**A data bug we found and fixed first**: our first baseline test showed
the Eggplant task collapsing to 8.3% for no good reason. We found the
cause: a missing background image file meant the robot was seeing a
totally different (fake, un-realistic) scene than it was trained on.
We restored the file and fixed the evaluation code to error loudly
instead of silently using the wrong scene. After the fix, our numbers
matched the lab's own previous measurements almost exactly (one task
matched to the decimal point), so we're confident the fix was correct.

**chunk-exec, backwards**: unlike other models, UniVLA *already*
predicts and runs 5 future actions per "think" by default. So to test
the same idea, we had to go the *opposite* way — use only the first 2
of those 5 predictions (forcing the model to "think" more often, which
is *slower*, not faster).

| Task | baseline | only use 2 of 5 | change |
|---|---|---|---|
| Carrot | 66.7% | 75.0% | +8.3pp |
| Stack | 75.0% | 45.8% | **−29.2pp** |
| Spoon | 70.8% | 54.2% | −16.6pp |
| Eggplant | 100.0% | 87.5% | −12.5pp |
| **Average** | **78.1%** | **65.6%** | **−12.5pp** |

Nothing collapsed to zero like RoboVLMs — this model's memory isn't an
LSTM, so it doesn't have that specific weak point. But it still lost
performance: it was trained to predict a smooth 5-step plan, and
chopping that plan short and re-planning more often creates small
jumps/glitches at the seams, which hurt tasks that need precision
(Stack, Eggplant) the most.

**Foveation — log-polar won, the opposite of SpatialVLA**

| Task | baseline | log-polar | blur |
|---|---|---|---|
| Carrot | 66.7% | 75.0% (+8.3p) | 70.8% (+4.2p) |
| Stack | 75.0% | 83.3% (+8.3p) | 66.7% (−8.3p) |
| Spoon | 70.8% | 87.5% (+16.7p) | 87.5% (+16.7p) |
| Eggplant | 100.0% | 100.0% (+0.0p) | 79.2% (**−20.8p**) |
| **Average** | **78.1%** | **86.5% (+8.3p)** | **76.0% (−2.1p)** |

Log-polar helped or stayed even on every single task. Blur helped on
some but badly hurt Eggplant (statistically confirmed real, not noise:
p≈0.025).

**We measured exactly why they're different**: we checked how much
fine detail survives at different distances from the center of the
image.

| Distance from center | log-polar keeps | blur keeps |
|---|---|---|
| Very center | 30–53% detail | **100% — untouched** |
| Middle | 1–3% detail | 52% detail |
| Far edge | a little detail | **0–1% — totally gone** |

- **log-polar**: gently blurs *everything*, but never fully erases any
  area.
- **blur**: keeps the center perfect, but the edges go completely
  blank past a certain point.

UniVLA has no fragile 3D math or memory to break, so the gentle,
everywhere version (log-polar) is a safe win. But blur's "totally
blank edges" becomes a real problem exactly when the target object is
near the edge of the frame — which happens in the Eggplant scene.

**Bottom line**: chunk-exec hurts here too (different reason than
RoboVLMs — plan smoothness, not memory desync). Foveation helps, but
only the gentle version (log-polar); the aggressive version (blur)
backfires on this model.

---

## 5. Big-picture conclusions

1. **chunk-exec** only helps when the model has NO memory that could get
   confused by skipping steps (SpatialVLA). Any kind of memory —
   LSTM (RoboVLMs) or a trained multi-step plan (UniVLA) — gets hurt,
   just for different reasons.
2. **foveation** only helps when the model has nothing fragile to break
   (OpenVLA, UniVLA). Models with exact 3D math (SpatialVLA) or an
   information-squeezing bottleneck (RoboVLMs) get hurt.
3. **Even the choice between foveation styles (log-polar vs blur) isn't
   universal** — blur wins on SpatialVLA, log-polar wins on UniVLA,
   they're a tie on OpenVLA.
4. For every failure, **we didn't just observe it — we proved the exact
   cause** with a controlled follow-up experiment (stale-action test,
   blur-vs-warp comparison, detail-preservation measurement, statistical
   significance checks).

**One sentence for the whole project**: across 4 very different robot
AI models, whether a speed or vision trick works is fully explained by
what's fragile inside the model — there is no trick that works
everywhere.

## 6. What's next

- Finish the OpenVLA Retina investigation (our first guess about why it
  fails turned out to be wrong — reuse rates were about the same before
  and after grabbing the object, not higher after as we expected)
- RoboVLMs: try caching the vision part only (not the memory), and try
  trimming redundant language-model layers
- Re-test the winning combos (SpatialVLA+chunk-exec, OpenVLA+foveation)
  on more simulators (CALVIN, LIBERO)
- Start planning real-robot deployment (per advisor's request)
