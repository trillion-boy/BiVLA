# Google Robot Tasks for Evaluation

## Available Tasks in SimplerEnv

SimplerEnv provides the following Google Robot tasks (from `simpler_env/__init__.py`):

### Pick Tasks (6 variants)
- `google_robot_pick_coke_can` (GraspSingleOpenedCokeCanInScene-v0)
- `google_robot_pick_horizontal_coke_can`
- `google_robot_pick_vertical_coke_can`
- `google_robot_pick_standing_coke_can`
- `google_robot_pick_object` (GraspSingleRandomObjectInScene-v0)

### Move Near Tasks (3 variants)
- `google_robot_move_near` (MoveNearGoogleBakedTexInScene-v1, default)
- `google_robot_move_near_v0` (MoveNearGoogleBakedTexInScene-v0)
- `google_robot_move_near_v1` (MoveNearGoogleBakedTexInScene-v1)

### Drawer Tasks (8 variants)
- `google_robot_open_drawer`, `_top_drawer`, `_middle_drawer`, `_bottom_drawer` (4 tasks)
- `google_robot_close_drawer`, `_top_drawer`, `_middle_drawer`, `_bottom_drawer` (4 tasks)

### Place Tasks (5 variants)
- `google_robot_place_in_closed_drawer` (OpenDrawerCustomInScene-v0)
- `google_robot_place_in_closed_top_drawer` (PlaceAppleInTopDrawerCustomInScene-v0)
- `google_robot_place_in_closed_middle_drawer`
- `google_robot_place_in_closed_bottom_drawer`
- `google_robot_place_apple_in_closed_top_drawer`

**Total: 21 tasks available**

---

## Recommended Subset for Evaluation

Choose a balanced subset covering different task families:

### Tier 1: Core Suite (4 tasks)
Mimics WidowX 4-task structure:
1. **Pick variant** → `google_robot_pick_coke_can` (grasp-only, like WidowX eggplant)
2. **Move variant** → `google_robot_move_near` (approach-only, like WidowX stack transport)
3. **Drawer variant** → `google_robot_open_drawer` (grasp + manipulation, like WidowX carrot)
4. **Place variant** → `google_robot_place_in_closed_drawer` (grasp + place, like WidowX spoon)

**Rationale:** Each covers different manipulation phases (grasp, move, open, place) and should show different adaptive chunk benefits.

### Tier 2: Extended Suite (6–8 additional tasks)
- Open/close subtasks: `open_top_drawer`, `open_middle_drawer`, `close_top_drawer` (3 tasks)
- Pick variants: `pick_object`, `pick_horizontal_coke_can` (2 tasks)
- Place variants: `place_in_closed_top_drawer` (1 task)

**Rationale:** Test generalization across task variants and difficulty levels.

---

## Task Definition Template

To add a task to `TASK_CONFIGS` in `spatialvla_eval.py`:

```python
"google_robot_pick_coke_can": {
    "env_name": "GraspSingleOpenedCokeCanInScene-v0",
    "robot": "google_robot",
    "scene_name": "???",  # Check env definition
    "rgb_overlay_path": None,  # Or path to Google Robot visual overlay if available
    "rgb_overlay_cameras": ["3rdperson_eye"],
    "obs_camera_name": "3rdperson_eye",
    "obj_init_options": ["random"],  # Grasp single object location
    "obj_episode_range": (0, 10),  # ~10 object spawns per task
    "max_episode_steps": 100,  # Adjust based on task complexity
    "robot_init_x": 0.0,
    "robot_init_y": 0.0,
    "control_freq": 10,  # Hz (Google Robot may differ from WidowX)
},
```

**Fields to determine per-task:**
- `env_name`: From SimplerEnv registry (e.g., "GraspSingleOpenedCokeCanInScene-v0")
- `scene_name`: Usually internal to env (may not be exposed, leave as template value)
- `rgb_overlay_path`: Check if Google Robot scene has a visual overlay image
- `obj_episode_range`: Number of object spawn configurations (typically 10–20)
- `max_episode_steps`: Task-dependent (pick ≈60–80, place ≈100–150)
- `robot_init_x`, `robot_init_y`: Workspace location for this task
- `control_freq`: Likely 10 Hz like WidowX, but verify

---

## Recommended Exploration Steps

### Step 1: Verify Task Availability

```bash
python -c "
import gymnasium as gym
from simpler_env.envs import *

tasks = ['google_robot_pick_coke_can', 'google_robot_move_near', 
         'google_robot_open_drawer', 'google_robot_place_in_closed_drawer']

for task in tasks:
    try:
        env = gym.make(f'{task}')
        print(f'✓ {task}')
        print(f'  Action space: {env.action_space}')
        print(f'  Obs keys: {env.observation_space.keys() if hasattr(env.observation_space, \"keys\") else \"N/A\"}')
        env.close()
    except Exception as e:
        print(f'✗ {task}: {e}')
"
```

### Step 2: Check Task Metadata

```bash
python -c "
from simpler_env.envs.google_robot_env import GoogleRobotEnv

# Inspect environment attributes
env = GoogleRobotEnv(task='pick_coke_can', render_mode=None)
print(f'Episode range: {getattr(env, \"obj_episode_range\", \"N/A\")}')
print(f'Max steps: {env._max_episode_steps}')
print(f'Control freq: {env.sim_freq} / {env.control_freq}')
env.close()
"
```

### Step 3: Run Single-Episode Pilot

```bash
python tome_spatialvla_eval.py \
    --model-path <ckpt> \
    --policy-setup google_robot \
    --task google_robot_pick_coke_can \
    --n-episodes 1 \
    --output-dir ./pilot_google_robot \
    2>&1 | head -100
```

Observe:
- Does it run without errors?
- What is the baseline success rate (expect ±50%, since Google Robot baseline not published)?
- What is model latency (should be ~900–1000 ms/step like SpatialVLA)?

---

## Baseline Expectations

### WidowX Baseline (for reference)
| Task | Success | Grasp | Latency |
|---|---|---|---|
| Eggplant | 66.7% | 87.5% | 836 ms |
| Carrot | 25.0% | 58.3% | ~900 ms |
| Spoon | 8.3% | 58.3% | ~900 ms |
| Stack | 31.3% | 58.3% | ~900 ms |

### Google Robot Expectations
Baseline success rates on Google Robot tasks are **not officially published** for SpatialVLA/UniVLA. Likely range:

| Task Family | Expected Success | Rationale |
|---|---|---|
| **Pick (grasp-only)** | 50–70% | Simpler than WidowX eggplant; fewer occlusions |
| **Move near** | 70–90% | Approach task, no grasp; easier than manipulation |
| **Drawer (open/close)** | 30–60% | Tool use; more complex than grasp-only |
| **Place** | 20–40% | Placement precision, hardest; similar to WidowX stack |

**Why different from WidowX?**
- Google Robot arm is different (longer reach, different control)
- Scenes may be easier/harder (more/less clutter)
- Task diversity (drawers add complexity)

---

## Using Google Robot Tasks in Evaluation

### Minimal Config (Add to `spatialvla_eval.py`)

```python
TASK_CONFIGS = {
    # ... existing WidowX tasks ...
    
    # Google Robot core suite (4 tasks)
    "google_robot_pick_coke_can": {
        "env_name": "GraspSingleOpenedCokeCanInScene-v0",
        "robot": "google_robot",
        # ... other fields (see template above)
    },
    "google_robot_move_near": {
        "env_name": "MoveNearGoogleBakedTexInScene-v1",
        "robot": "google_robot",
        # ...
    },
    # ... etc
}
```

### Run Comparison

```bash
# Run fixed k=4 on Google Robot tasks
for task in google_robot_pick_coke_can google_robot_move_near \
            google_robot_open_drawer google_robot_place_in_closed_drawer; do
    python tome_spatialvla_eval.py --model-path <ckpt> --task $task \
        --exec-chunk 4 --n-episodes 24 --output-dir results/google_robot_k4/$task
done

# Then run adaptive
for task in google_robot_pick_coke_can google_robot_move_near \
            google_robot_open_drawer google_robot_place_in_closed_drawer; do
    python tome_spatialvla_eval_adaptive.py --model-path <ckpt> --task $task \
        --adaptive-chunk --n-episodes 24 --output-dir results/google_robot_adaptive/$task
done
```

---

## Integration Checklist

- [ ] Determine env_name for each task (from SimplerEnv registry)
- [ ] Determine scene_name and init positions
- [ ] Determine obj_episode_range (number of object spawns)
- [ ] Determine max_episode_steps per task
- [ ] Run pilot on 1 task (pick_coke_can) to validate integration
- [ ] If successful, add all 4 core tasks to TASK_CONFIGS
- [ ] Run baseline on all 4 tasks (N=8 each for speed, then N=24)
- [ ] Run k=4 and adaptive on all 4 tasks
- [ ] Consolidate results in cross-robot comparison table

---

## References

- SimplerEnv tasks: `SimplerEnv/simpler_env/__init__.py`
- Google Robot robot definition: `SimplerEnv/ManiSkill2_real2sim/.../googlerobot.py`
- WidowX task template: `SpatialVLA/experiments/latent_saccade/spatialvla_eval.py` (TASK_CONFIGS)
