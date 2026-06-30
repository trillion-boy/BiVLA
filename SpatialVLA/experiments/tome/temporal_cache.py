"""
Temporal caching for SpatialVLA's frozen SigLIP encoder (frame-stride variant).

The profiler showed encoding (SigLIP ViT) is ~14% of a control step. Consecutive
control frames are highly similar (the arm moves a little; the background is
static), so we can **reuse the previous frame's image features** for a few steps
and recompute the ViT only every `stride` steps. This skips ~(stride-1)/stride of
the encoding cost — training-free, no architecture change.

This is the simplest, cleanest temporal-caching variant: the ViT mixes all
patches via global attention, so per-patch reuse isn't possible; whole-feature
frame-stride reuse is. Tradeoff: reused features are stale by up to `stride-1`
steps, which can hurt precise moments (grasp) — so keep `stride` small (2-3) and
measure success.

Usage:
    apply_temporal_cache(policy.vla, stride=2)
    ...
    reset_temporal_cache(policy.vla)   # at the start of each episode
"""
from __future__ import annotations


def apply_temporal_cache(model, stride: int = 2, verbose: bool = True):
    """Patch `model.get_image_features` to recompute the ViT only every `stride`
    steps and reuse the cached features in between. Call `reset_temporal_cache`
    per episode so the first step always recomputes."""
    if getattr(model, "_tc_patched", False):
        remove_temporal_cache(model)
    model._tc_orig_gif = model.get_image_features
    model._tc_stride = max(1, int(stride))
    model._tc_counter = 0
    model._tc_cache = None

    def cached_gif(pixel_values, intrinsic, *args, **kwargs):
        # recompute on the first call of an episode and every `stride` calls
        if model._tc_cache is None or (model._tc_counter % model._tc_stride == 0):
            model._tc_cache = model._tc_orig_gif(pixel_values, intrinsic, *args, **kwargs)
        model._tc_counter += 1
        return model._tc_cache

    model.get_image_features = cached_gif
    model._tc_patched = True
    if verbose:
        save = (stride - 1) / stride
        print(f"[TemporalCache] reuse SigLIP features, recompute every {stride} steps "
              f"(~{save:.0%} of encoding skipped; encoding≈14% of step).", flush=True)
    return model


def reset_temporal_cache(model):
    """Clear the cache and counter so the next step recomputes (call per episode)."""
    if getattr(model, "_tc_patched", False):
        model._tc_counter = 0
        model._tc_cache = None
    return model


def remove_temporal_cache(model):
    if getattr(model, "_tc_patched", False):
        model.get_image_features = model._tc_orig_gif
        del model._tc_orig_gif
        model._tc_patched = False
        model._tc_cache = None
    return model
