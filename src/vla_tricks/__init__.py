"""Reusable training-free interventions for OpenVLA experiments."""

from .depth import StaticDepthPruner, collect_block_influence, select_non_adjacent
from .foveation import foveate_blur
from .perception import (
    InteractionAwareTemporalFusion,
    PatchReuseDecision,
    fuse_projected_tokens,
    patch_entropy,
    patch_motion,
    select_reusable_patches,
)
from .temporal import ConservativeActionReuse, apply_action_repeat
from .vla_cache import cache_positions_from_decision, visual_task_relevance

__all__ = [
    "ConservativeActionReuse",
    "InteractionAwareTemporalFusion",
    "PatchReuseDecision",
    "StaticDepthPruner",
    "apply_action_repeat",
    "cache_positions_from_decision",
    "collect_block_influence",
    "foveate_blur",
    "fuse_projected_tokens",
    "patch_entropy",
    "patch_motion",
    "select_reusable_patches",
    "select_non_adjacent",
    "visual_task_relevance",
]
