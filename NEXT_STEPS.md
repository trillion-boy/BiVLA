# Next Steps — Implementation Status & Plan

**Date:** July 2026  
**Author:** Junseo  
**Session:** claude/serene-davinci-sy33re

---

## Summary of Recent Work

### 1. Adaptive Chunk Execution ✅ COMPLETE

**What was implemented:**
- `AdaptiveChunkExecutor` class: switches k based on gripper state
- Phase detection: k=4 (transport) ↔ k=1 (grasp/place)
- Configurable delays: `close_steps_before_dense`, `open_steps_after_release`
- Integration into `tome_spatialvla_eval_adaptive.py`
- Unit tests covering all phase transitions

**Key files:**
- `SpatialVLA/experiments/tome/adaptive_chunk_exec.py` (implementation)
- `SpatialVLA/experiments/tome/test_adaptive_chunk_exec.py` (tests, all passing)
- `SpatialVLA/experiments/tome/tome_spatialvla_eval_adaptive.py` (evaluation)
- `SpatialVLA/experiments/tome/ADAPTIVE_CHUNK_README.md` (docs)

**Expected behavior:**
- k=4 during transport → 3.6× speed
- k=1 during grasp/place → baseline latency (precision)
- Per-episode speedup: **1.5–1.8×** (faster than k=2's 1.9× but fixes stack task regression)
- Expected success rates: +13pp maintained while recovering placement accuracy

---

## Immediate Next Steps (This Week)

### Priority 1: Run Adaptive Chunk Experiments

**Current blocking issue:** Need access to SimplerEnv/SpatialVLA eval environment with GPU

**Tasks:**
1. **Validate adaptive on all 4 WidowX tasks** (N=24 per task)
   - Test with default params: k_sparse=4, k_dense=1, delays=3,5
   - Compare vs: baseline (k=1), fixed k=2, fixed k=4
   - Metric: success rate, grasp rate, ms/step

2. **Hyperparameter sweep** (if results justify)
   - close_delay: {1, 2, 3, 5}
   - open_delay: {3, 5, 7, 10}
   - Find optimal for each task or one "robust" config

3. **Per-task analysis**
   - For each task, report: success rate, grasp rate, avg steps, ms/step
   - Failure breakdown: never-grasped, grasped-but-not-placed, etc.
   - Identify if stack task (placement-critical) improves with adaptive

**Expected outcome:**
- Stack success: 25% (k=4) → 31%+ (adaptive) [recover to baseline]
- Mean success: 45.9% (k=4) → 46%+ (adaptive) [hold improvement]
- Mean latency: 50.8% (k=4) → 55%+ (adaptive) [trade ~400ms for accuracy]

---

### Priority 2: Expand to New Tasks

**Why:** Current 4 WidowX tasks are not sufficient for paper generality claims

**Plan:**
1. **Identify Google Robot tasks** (preferred, already checkpoint-compatible)
   - Search `SimplerEnv/` for google_robot scene definitions
   - Expected: 2–4 tasks (reach, grasp, place variants)
   
2. **Define task configs** following widowx template
   - Add to `TASK_CONFIGS` in `spatialvla_eval.py` or new task registry
   - Verify obj_episode_range, init positions, max_steps

3. **Run baseline on Google Robot** (quick validation)
   - Establish baseline success rates on new robot
   - Confirm checkpoint works without errors

4. **Run adaptive chunk on Google Robot** (full sweep)
   - Same config as WidowX or per-robot tuned?
   - Merge results into consolidated table

**Expected outcome:**
- At least 4–6 new tasks from different robot
- Cross-robot validation: k_sparse/k_dense work universally?
- Paper claims: "non-uniform allocation effective on WidowX and Google Robot, likely generalizes"

**Timeline:** 1–2 weeks depending on task availability

---

### Priority 3: Integrate RetinaBased / Mentor's OpenVLA Work

**What is RetinaBased?** Mentor's independent implementation of foveation + action reuse for OpenVLA
- **Approach:** Log-polar visual transform (pixel-level, non-uniform, keep only fovea) + skip model when motion low
- **Results:** 3.1% → 32.3% success on 4 WidowX (but model is OpenVLA, different backbone than SpatialVLA)
- **Complementary:** Our chunk execution is temporal (call model less), theirs is spatial+temporal

**Next actions:**
1. **Set up RetinaBased environment locally**
   - Clone mentor's repo (need URL)
   - Set up Colab + local SimplerEnv (mentor's repo description says both available)
   - Run mentor's eval to reproduce 32.3% result

2. **Compare implementations**
   - Our approach: k=4 (execute predicted chunk) + adaptive k (phase-aware)
   - Their approach: log-polar visual + action reuse (grasp-triggered frame skip)
   - Both achieve +speed, +success on same baseline (3.1%)
   
3. **Potential combination** (longer term)
   - Mentor's: spatial non-uniformity (sharp center, blurred periphery)
   - Ours: temporal non-uniformity (sparse replan + phase-aware density)
   - Together: full non-uniform allocation (spatial + temporal)
   - Expected: more than additive gains? (need measurement)

**Blocker:** Need mentor's RetinaBased repo URL (likely in previous conversation or shared externally)

---

## Medium-Term Plan (2–3 Weeks)

### Phase 1: Full Results Table

**Consolidate:**
```
Model               | Task          | Baseline | k=2   | k=4   | Adaptive | Speedup | Notes
SpatialVLA (ours)   | eggplant      | 66.7%    | 87.5% | 87.5% | 87.5%+   | 1.5–2× | ✓ transport benefit
SpatialVLA (ours)   | carrot        | 25.0%    | 41.7% | 41.7% | 40%+     | 1.5–2× | ✓ approach benefit
SpatialVLA (ours)   | spoon         | 8.3%     | 29.2% | 29.2% | 28%+     | 1.5–2× | ✓ transport benefit
SpatialVLA (ours)   | stack         | 31.3%    | 31.3% | 25.0% | 31%+     | 1.0–1.5× | GOAL: adaptive recovers
SpatialVLA (new)    | google_*      | ~40%?    | ↑10%? | ±3%?  | ↑?       | 1.5–2× | (TBD)
UniVLA (k-sweep)    | eggplant      | 100%     | ↓2%   | ↓5%   | (TBD)    | (TBD)  | (later sweep)
OpenVLA (mentor)    | eggplant      | 3.1%     | →32%  | -     | -        | 0.89×  | foveation+reuse
```

**Roadmap:**
1. Run adaptive on WidowX (this week)
2. Run adaptive on Google Robot (if available next week)
3. Run UniVLA k-sweep (if resources + time permit)
4. Mentor's OpenVLA: use for paper narrative but not our own experiments

### Phase 2: Paper Draft

**Structure (likely):**
1. **Introduction:** non-uniform resource allocation (spatial, temporal, depth axes)
2. **Background:** SpatialVLA's decode-bound latency (75% per step)
3. **Our contribution:** temporal non-uniformity via action chunking + phase-aware k
4. **Experiments:**
   - Phase 1: fixed k (2, 4) on WidowX (published, in progress report)
   - Phase 2: adaptive k on WidowX (in progress, this session)
   - Phase 3: new tasks (Google Robot if available)
   - Phase 4: UniVLA k-sweep (optional, lower priority)
5. **Mentor's work (acknowledgment/discussion):**
   - Independent arrival at "call model less often" lever
   - Complementary approach (spatial vs. temporal)
   - Potential for combined methods

**Key claims:**
- On frozen SpatialVLA: adaptive k recovers placement accuracy while maintaining 1.5–1.8× speedup
- Generalizes to new tasks (if Google Robot shows similar gains)
- Non-uniform allocation is a principled framework (spatial, temporal, depth axes)

---

## Optional / Future

### 1. Robot Hand Experiments

**Status:** Mentor mentioned potential robot hand deployment assistance  
**Constraint:** Real-world evaluation is beyond scope of this session  
**Action:** Document plan, defer until after sim results finalize

### 2. UniVLA k-Sweep (Lower Priority)

**Task:** Evaluate UniVLA with k=1,2,3,4,5 to show replan-frequency curve  
**Why lower priority?**
- Already have depth-pruning results on UniVLA (1.1× on eggplant)
- k-sweep less impactful than adaptive (UniVLA hit FFN-bound on decode, not decode count)
- Resources better spent on new tasks + adaptive validation

**If time permits:** Run k=2 and k=4 on eggplant (N=24) to show UniVLA doesn't benefit as much

### 3. Mentor Collaboration

**Potential (post-session):**
- Integrate RetinaBased evaluation into BiVLA eval suite
- Cross-validate adaptive chunk on mentor's OpenVLA (same baseline, different model)
- Joint paper sections if timeline aligns

---

## Blockers & Risks

| Issue | Impact | Mitigation |
|---|---|---|
| No eval environment access this week | Can't run adaptive experiments | Wait for user's Colab/local setup, or run on next session |
| Google Robot tasks not in SimplerEnv | Can't expand beyond WidowX | Check UniVLA paper, search SimplerEnv codebase, or define custom scenes |
| RetinaBased repo not accessible | Can't integrate mentor's work | Request URL from mentor or find in previous conversation logs |
| Adaptive params not optimal | Results don't match expected gains | Hyperparameter sweep if time (close_delay, open_delay) |
| Stack task still fails at adaptive | Hypothesis (grasp precision) wrong | Fall back to fixed k=2 (which worked), investigate root cause further |

---

## Summary of Deliverables This Session

✅ **Implemented:**
1. AdaptiveChunkExecutor class + tests
2. Integration into evaluation script
3. Documentation (README, next steps)
4. Expansion plan for new tasks

⏳ **In Progress (Blocked on Eval Environment):**
1. Run adaptive on 4-task WidowX suite
2. Find & run Google Robot tasks
3. Consolidate results table

📋 **To Do (Next Sessions):**
1. Integrate RetinaBased / OpenVLA results
2. Finalize paper draft
3. Potential robot hand experiments

---

## Key Metrics to Track

**For each task / config:**
- Success rate (primary metric)
- Grasp rate (diagnostic)
- Mean steps (episode length)
- **ms/step** (latency, model-time only)
- **ms/total** (wall-clock, for reality check)

**Aggregates:**
- Mean success (across 4 tasks)
- Mean grasp (diagnostic for phase detection)
- Mean latency (how well adaptive trades speed for accuracy)

**Failure analysis:**
- Never-grasped → approach problem
- Grasped-but-not-placed → place precision problem
- Success rate by phase (if instrumentable)

---

## Running the Experiments

### Adaptive Chunk (when environment available)

```bash
# Baseline
python tome_spatialvla_eval.py --model-path <ckpt> --task widowx_stack_cube \
  --n-episodes 24 --output-dir results/adaptive_baseline

# Fixed k=2
python tome_spatialvla_eval.py --model-path <ckpt> --task widowx_stack_cube \
  --exec-chunk 2 --n-episodes 24 --output-dir results/adaptive_k2

# Fixed k=4
python tome_spatialvla_eval.py --model-path <ckpt> --task widowx_stack_cube \
  --exec-chunk 4 --n-episodes 24 --output-dir results/adaptive_k4

# Adaptive (default params)
python tome_spatialvla_eval_adaptive.py --model-path <ckpt> --task widowx_stack_cube \
  --adaptive-chunk --n-episodes 24 --output-dir results/adaptive_adaptive

# Adaptive (tuned params)
python tome_spatialvla_eval_adaptive.py --model-path <ckpt> --task widowx_stack_cube \
  --adaptive-chunk --adaptive-close-delay 2 --adaptive-open-delay 7 \
  --n-episodes 24 --output-dir results/adaptive_tuned
```

---

## Questions for Next Session

1. **Environment setup:** When can we run experiments? (Colab? Local?)
2. **Google Robot tasks:** Are they available in SimplerEnv? (Search needed)
3. **RetinaBased access:** What's the URL? (Mentor's repo)
4. **Paper timeline:** When is draft due? (Affects priority)
5. **Real robot:** Interest in hand experiments? (Scope + timeline)
