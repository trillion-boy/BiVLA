import numpy as np
import pytest
import torch

from vla_tricks.depth import StaticDepthPruner, collect_block_influence, select_non_adjacent
from vla_tricks.foveation import foveate_blur
from vla_tricks.perception import (
    InteractionAwareTemporalFusion,
    fuse_projected_tokens,
    patch_entropy,
    patch_motion,
    select_reusable_patches,
)
from vla_tricks.temporal import ConservativeActionReuse, apply_action_repeat
from vla_tricks.vla_cache import (
    cache_positions_from_decision,
    task_relevant_static_tokens,
    visual_task_relevance,
)


def test_foveation_preserves_shape_dtype_and_centre():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    output = foveate_blur(image, 0.2)
    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert np.array_equal(output[60:68, 60:68], image[60:68, 60:68])
    assert not np.array_equal(output[:16, :16], image[:16, :16])


def test_action_repeat_order():
    actions = np.array([[1], [2], [3]])
    assert apply_action_repeat(actions, 2).ravel().tolist() == [1, 1, 2, 2, 3, 3]


def test_conservative_reuse_observes_and_limits_open_loop_horizon():
    controller = ConservativeActionReuse(
        max_frame_mae=0.001,
        min_action_cosine=0.99,
        min_translation_norm=0.01,
        max_consecutive_reuse=1,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    calls = 0

    def infer():
        nonlocal calls
        calls += 1
        return np.array([0.05, 0, 0, 0, 0, 0, 1], dtype=np.float32)

    reused = [controller.step(image, infer)[1] for _ in range(4)]
    assert reused == [False, False, True, False]
    assert calls == 3

    changed = image.copy()
    changed[:, :, :] = 255
    assert controller.step(changed, infer)[1] is False


def test_reuse_is_disabled_for_fine_motion_and_gripper_transition():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    actions = iter(
        [
            np.array([0.001, 0, 0, 0, 0, 0, -1]),
            np.array([0.001, 0, 0, 0, 0, 0, -1]),
            np.array([0.05, 0, 0, 0, 0, 0, 1]),
        ]
    )
    controller = ConservativeActionReuse(min_translation_norm=0.01)
    assert not controller.step(image, lambda: next(actions))[1]
    assert not controller.step(image, lambda: next(actions))[1]
    assert not controller.step(image, lambda: next(actions))[1]


def test_reuse_rejects_small_local_change_hidden_by_global_average():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    controller = ConservativeActionReuse(
        max_frame_mae=0.02,
        max_local_patch_mae=0.1,
        local_grid_size=8,
        signature_stride=1,
        min_translation_norm=0.01,
    )
    action = np.array([0.05, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    assert not controller.step(image, lambda: action)[1]
    assert not controller.step(image, lambda: action)[1]

    changed = image.copy()
    changed[:8, :8] = 255
    assert float(np.abs(changed.astype(float) - image).mean() / 255.0) < 0.02
    assert not controller.step(changed, lambda: action)[1]


def test_patch_signals_and_interaction_aware_selection():
    previous = np.zeros((64, 64, 3), dtype=np.uint8)
    current = previous.copy()
    # Patch 0 changes, patch 5 is textured, and patch 10 is task-relevant.
    current[:16, :16] = 255
    checkerboard = (np.indices((16, 16)).sum(axis=0) % 2 * 255).astype(np.uint8)
    current[16:32, 16:32] = checkerboard[..., None]
    relevance = np.zeros(16, dtype=np.float32)
    relevance[10] = 1.0

    motion = patch_motion(previous, current, grid_size=(4, 4))
    entropy = patch_entropy(current, grid_size=(4, 4))
    assert motion.argmax() == 0
    assert entropy.argmax() == 5

    decision = select_reusable_patches(
        previous,
        current,
        task_relevance=relevance,
        grid_size=(4, 4),
        motion_threshold=0.01,
        entropy_protect_fraction=1 / 16,
        task_protect_fraction=1 / 16,
        protect_radius=0,
        max_reuse_fraction=0.5,
    )
    assert {0, 5, 10}.issubset(decision.protected_patch_ids)
    assert not {0, 5, 10}.intersection(decision.reusable_patch_ids)
    assert len(decision.reusable_patch_ids) == 8
    assert cache_positions_from_decision(decision) == [
        index + 1 for index in decision.reusable_patch_ids
    ]


def test_temporal_fusion_hook_reuses_only_selected_tokens_and_refreshes():
    previous = np.zeros((32, 32, 3), dtype=np.uint8)
    current = previous.copy()
    current[:8, :8] = 255
    projector = torch.nn.Identity()
    controller = InteractionAwareTemporalFusion(
        keyframe_interval=2,
        grid_size=(4, 4),
        motion_threshold=0.01,
        entropy_protect_fraction=0.0,
        task_protect_fraction=0.0,
        protect_radius=0,
        max_reuse_fraction=0.5,
    )
    controller.attach(projector)
    try:
        controller.prepare(previous)
        first = projector(torch.zeros(1, 16, 2))
        decision = controller.prepare(current)
        assert decision is not None and 0 in decision.protected_patch_ids
        second = projector(torch.ones(1, 16, 2))
        assert torch.equal(second[:, 0], torch.ones(1, 2))
        for index in decision.reusable_patch_ids:
            assert torch.equal(second[:, index], first[:, index])

        # Interval two forces the third observation to be a dense keyframe.
        assert controller.prepare(current) is None
        third = projector(torch.full((1, 16, 2), 2.0))
        assert torch.equal(third, torch.full((1, 16, 2), 2.0))
    finally:
        controller.detach()


def test_fuse_projected_tokens_validates_and_preserves_unselected_tokens():
    previous = torch.zeros(1, 4, 2)
    current = torch.ones(1, 4, 2)
    fused = fuse_projected_tokens(current, previous, [1, 3])
    assert fused[:, [1, 3]].sum() == 0
    assert fused[:, [0, 2]].sum() == 4


class _Layer(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states):
        direction = torch.arange(
            hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device
        )
        direction = direction - direction.mean()
        return (hidden_states + self.scale * direction,)


def test_influence_averages_calibration_runs():
    layers = torch.nn.ModuleList([_Layer(0.001), _Layer(0.5)])

    def run(value):
        hidden = torch.full((1, 3, 4), value)
        for layer in layers:
            hidden = layer(hidden)[0]

    scores = collect_block_influence(layers, [lambda: run(0.2), lambda: run(1.0)])
    assert len(scores) == 2
    assert scores[0] < scores[1]


def test_layer_selection_protects_ends_and_strictly_enforces_gap():
    influence = [10, 9, 0.4, 0.1, 0.3, 0.2, 0.0]
    chosen = select_non_adjacent(
        influence, 2, min_layer_fraction=0.25, protect_last=1, min_gap=1
    )
    assert 0 not in chosen and 1 not in chosen and 6 not in chosen
    assert abs(chosen[0] - chosen[1]) > 1
    with pytest.raises(ValueError):
        select_non_adjacent(influence, 4, protect_last=1, min_gap=1)


class _Attention:
    def __init__(self, index):
        self.layer_idx = index


class _Decoder(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.self_attn = _Attention(index)


class _NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Decoder(i) for i in range(6)])
        self.config = type("Config", (), {"num_hidden_layers": 6})()


def test_structural_depth_pruner_reindexes_and_restores():
    model = _NestedModel()
    originals = list(model.layers)
    pruner = StaticDepthPruner(model)
    pruner.apply((2, 4))
    assert list(model.layers) == [originals[i] for i in (0, 1, 3, 5)]
    assert [layer.self_attn.layer_idx for layer in model.layers] == [0, 1, 2, 3]
    assert model.config.num_hidden_layers == 4
    pruner.restore()
    assert list(model.layers) == originals
    assert [layer.self_attn.layer_idx for layer in model.layers] == list(range(6))
    assert model.config.num_hidden_layers == 6


def test_vla_cache_task_relevance_excludes_important_static_tokens():
    attention = torch.zeros(1, 2, 260, 260)
    # Text queries strongly attend to visual patch 4 (cache position 5).
    attention[:, :, 257:, 5] = 10.0
    reusable = task_relevant_static_tokens(
        [attention], [0, 4, 8], layer_id=0, top_k=1
    )
    assert reusable == [1, 9]
    relevance = visual_task_relevance([attention], layer_id=0)
    assert relevance.argmax().item() == 4
