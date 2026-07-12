"""Adaptive chunk execution: varies k based on gripper state and manipulation phase.

Instead of fixed k (replanning every 1/k steps), adapt k based on phase:
  - Transport/approach: k=4 (sparse, ~3.6x speed, motion-heavy)
  - Near grasp/place: k=1 (dense, baseline speed, precision-critical)

Detects phase by tracking gripper opening/closing and per-step feedback.
"""

from chunk_exec import apply_chunk_execution, reset_chunk_execution, remove_chunk_execution


class AdaptiveChunkExecutor:
    """Wraps chunk_exec to vary k based on gripper state."""

    def __init__(self, policy, k_sparse=4, k_dense=1,
                 gripper_close_threshold=0.3,
                 close_steps_before_dense=3,
                 open_steps_after_release=5):
        """
        Args:
            policy: frozen VLA policy
            k_sparse: chunk size during transport/approach
            k_dense: chunk size during grasp/place (precision-critical)
            gripper_close_threshold: gripper value ≤ this means gripper is closed
            close_steps_before_dense: steps after gripper closes before switching to k=1
            open_steps_after_release: steps after gripper opens before switching back to k=4
        """
        self.policy = policy
        self.k_sparse = k_sparse
        self.k_dense = k_dense
        self.gripper_close_threshold = gripper_close_threshold
        self.close_steps_before_dense = close_steps_before_dense
        self.open_steps_after_release = open_steps_after_release

        # State tracking
        self.current_k = k_sparse
        self.current_chunk_state = None
        self.gripper_closed = False
        self.steps_since_gripper_change = 0
        self.episode_step = 0

        # Apply initial sparse chunk execution
        self._set_chunk_k(k_sparse)

    def _set_chunk_k(self, k):
        """Switch chunk execution to new k value."""
        if k == self.current_k:
            return

        # If chunk execution is already active, remove it
        if self.current_chunk_state is not None:
            remove_chunk_execution(self.policy)

        # Apply new chunk execution with new k
        self.current_chunk_state = apply_chunk_execution(self.policy, k=k, verbose=False)
        self.current_k = k

    def step(self, action):
        """Call after each action step. Updates k based on gripper state."""
        self.episode_step += 1

        # Extract gripper value from action
        gripper_val = action.get("gripper", 1.0)  # 1.0 = open, ≤0.3 = closed

        # Detect gripper state change
        prev_gripper_closed = self.gripper_closed
        self.gripper_closed = gripper_val <= self.gripper_close_threshold

        if self.gripper_closed != prev_gripper_closed:
            # Gripper state changed
            self.steps_since_gripper_change = 0
        else:
            self.steps_since_gripper_change += 1

        # Decide k based on phase
        if self.gripper_closed:
            # Gripper is closed: we're grasping/placing
            # After enough steps of closure, switch to dense
            if self.steps_since_gripper_change >= self.close_steps_before_dense:
                target_k = self.k_dense
            else:
                # Just closed; stay sparse for a bit longer
                target_k = self.current_k
        else:
            # Gripper is open: we're transporting/approaching
            # After enough steps of release, switch back to sparse
            if self.steps_since_gripper_change >= self.open_steps_after_release:
                target_k = self.k_sparse
            else:
                # Just opened; stay at current k for a bit longer
                target_k = self.current_k

        # Apply k change if needed
        if target_k != self.current_k:
            self._set_chunk_k(target_k)

    def reset(self):
        """Call at episode reset to flush the queue."""
        reset_chunk_execution(self.policy)
        self.gripper_closed = False
        self.steps_since_gripper_change = 0
        self.episode_step = 0
        # Reset to sparse k for new episode
        if self.current_k != self.k_sparse:
            self._set_chunk_k(self.k_sparse)

    def remove(self):
        """Remove chunk execution wrapper."""
        if self.current_chunk_state is not None:
            remove_chunk_execution(self.policy)
            self.current_chunk_state = None
            self.current_k = None

    def get_state(self):
        """Return current state for logging."""
        return {
            "episode_step": self.episode_step,
            "current_k": self.current_k,
            "gripper_closed": self.gripper_closed,
            "steps_since_change": self.steps_since_gripper_change,
            "chunk_state": self.current_chunk_state,
        }
