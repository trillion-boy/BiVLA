# Expanding Evaluation to New Tasks

## Current Status

Current evaluation is limited to **4 WidowX Bridge tasks**:
- `widowx_put_eggplant_in_basket`
- `widowx_carrot_on_plate`
- `widowx_stack_cube`
- `widowx_spoon_on_towel`

**Why expand?** To demonstrate that:
1. Adaptive chunk execution generalizes beyond WidowX
2. Non-uniform resource allocation works across robot morphologies
3. Results aren't specific to a particular task distribution

## Candidate Sources

### 1. UniVLA Paper Tasks (Primary Source)

UniVLA paper evaluates on:
- **WidowX Bridge** (4 tasks, currently covered)
- **Google Robot** (2–4 tasks)
- **Franka** (if available)
- **Real robot deployment** (out of scope for sim-only now)

**Action:** Check UniVLA paper appendix / supplementary materials for task definitions and corresponding SimplerEnv environments.

### 2. SimplerEnv Available Tasks

SimplerEnv (the underlying environment library) likely supports additional tasks beyond the 4 WidowX ones. Need to:
1. Check `SimplerEnv/` directory for task definitions
2. Look for `*Scene-v0` environment registrations
3. Identify which are:
   - Stable in simulation (no physics instability)
   - Relevant to manipulation (grasp + place or grasp + tool use)
   - Covered by a pre-trained VLA policy

**Command to explore:**
```bash
python -c "import gymnasium as gym; print([k for k in gym.envs.registry.keys() if 'Scene' in k])"
```

### 3. Prior Work

Related papers to check:
- **AutoRT**: evaluates on Google Robot tasks
- **VLA-Rope**: extends to deformable object manipulation
- **ViLA** / **Octo**: may have task suites compatible with SimplerEnv

## Proposed Expansion Plan

### Phase 1: Google Robot (Recommended First)

**Why:** Already supported by SpatialVLA checkpoint (model handles both widowx_bridge and google_robot setups).

**Expected tasks:**
- `google_robot_move_to_drawer`
- `google_robot_pick_from_drawer`
- `google_robot_reach_to_object`
- Similar 2–4 task suite

**Implementation:**
1. Define task configs (like widowx tasks) in `spatialvla_eval.py`
2. Register in `TASK_CONFIGS`
3. Run baseline + adaptive chunk sweep
4. Compare vs WidowX results (task-independent vs task-specific improvements?)

### Phase 2: Additional WidowX Variants (Optional)

If Google Robot has limited tasks:
- `widowx_insert_obj_into_container`
- `widowx_knock_object_off_surface`
- Other WidowX Bridge scenarios

**Check:** SimplerEnv codebase for available scenes.

### Phase 3: Cross-Backbone Validation

After evaluating adaptive chunk on all SpatialVLA tasks:
1. Run UniVLA k-sweep (k=1..5) on subset of new tasks
2. Compare Pareto frontier: k=2 vs adaptive vs k=4 across backbones
3. Create consolidated table for paper

## Action Items

1. **Find UniVLA paper task definitions**
   - If paper has task names, search SimplerEnv for matching scenes
   - If paper only has results, infer from task counts (n=4 for WidowX)

2. **Explore SimplerEnv tasks**
   - `grep -r "Scene" SimplerEnv/` to find registered environments
   - Check which require pre-trained checkpoints vs. are generic

3. **Verify google_robot checkpoint support**
   - Test: `python tome_spatialvla_eval.py --policy-setup google_robot --task <new_task>`
   - Confirm checkpoint loads without errors

4. **Define task configs**
   - Follow template from widowx tasks (env_name, robot, scene_name, etc.)
   - Verify obj_episode_range, init positions are correct

5. **Run pilot experiments**
   - Baseline + adaptive on 1–2 new tasks to validate integration
   - Then sweep full suite if stable

## Expected Timeline

- **Week 1:** Find & define new tasks (Google Robot preferred)
- **Week 2:** Run baseline + k=4 on new tasks (quick scan)
- **Week 3:** Run full adaptive chunk sweep on all new tasks
- **Week 4:** Consolidate results + prepare paper draft

## Success Criteria

- [ ] At least 2–4 tasks from a different robot/environment
- [ ] Adaptive chunk works without crashing on new tasks
- [ ] Success rates similar in scale (not requiring full re-tuning)
- [ ] Speedups hold (1.5–1.8× for adaptive, 1.9× for k=2)
- [ ] Mean improvement ≥ +10pp (more lenient than WidowX +13pp if new tasks are harder)

## Notes

- **Checkpoint compatibility:** SpatialVLA checkpoint should handle different robots, but confirm
- **Distribution shift:** New tasks may have different difficulty; establish per-task baselines
- **Phase detection:** Adaptive gripper detection should generalize (all robots have gripper state)
- **Chunk size:** If new robot uses different action chunk size (e.g., 3 instead of 4), need config option
