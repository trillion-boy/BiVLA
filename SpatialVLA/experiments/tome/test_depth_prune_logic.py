"""CPU unit tests for DepthPruner (no SpatialVLA / GPU needed)."""
import torch
import torch.nn as nn

from depth_prune_gemma2 import DepthPruner, _BypassLayer


class FakeLayer(nn.Module):
    """A toy Gemma2-style decoder layer: adds a learnable-ish delta, returns tuple."""
    def __init__(self, delta_scale):
        super().__init__()
        self.delta_scale = delta_scale

    def forward(self, hidden_states, *a, **k):
        # bigger delta_scale => output diverges more from input => LESS redundant
        return (hidden_states + self.delta_scale * torch.ones_like(hidden_states),)


class FakeGemma2Model(nn.Module):
    def __init__(self, deltas):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(d) for d in deltas])

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)[0]
        return h


class FakeLM(nn.Module):
    def __init__(self, deltas):
        super().__init__()
        self.model = FakeGemma2Model(deltas)


def test_bypass_is_identity():
    x = torch.randn(2, 3, 4)
    b = _BypassLayer(FakeLayer(9.0))
    out = b(x)[0]
    assert torch.equal(out, x), "bypass must return input unchanged"
    print("ok: bypass is identity")


def test_ranking_orders_by_redundancy():
    # layer 2 has the smallest delta -> input≈output -> most redundant -> ranked first
    deltas = [1.0, 0.5, 0.01, 0.8]
    lm = FakeLM(deltas)
    p = DepthPruner(lm)
    p.install_calibration_hooks()
    lm.model(torch.randn(1, 5, 8))
    ranking = p.finalize_calibration()
    assert ranking[0] == 2, f"most-redundant should be layer 2, got {ranking}"
    print(f"ok: ranking most->least redundant = {ranking}  (redundancy={[round(x,3) for x in p._redundancy]})")


def test_apply_and_restore():
    deltas = [1.0, 0.5, 0.01, 0.02, 0.8, 0.9]
    lm = FakeLM(deltas)
    p = DepthPruner(lm)
    p.install_calibration_hooks()
    lm.model(torch.randn(1, 5, 8))
    p.finalize_calibration()

    pruned = p.apply(count=2, min_layer=1)
    # layers 2 and 3 are most redundant (0.01, 0.02), both allowed (>=min_layer=1, not last)
    assert set(pruned) == {2, 3}, f"expected {{2,3}}, got {pruned}"
    assert isinstance(lm.model.layers[2], _BypassLayer)
    assert isinstance(lm.model.layers[3], _BypassLayer)

    p.restore()
    assert not isinstance(lm.model.layers[2], _BypassLayer), "restore must put real layers back"
    assert isinstance(lm.model.layers[2], FakeLayer)
    print(f"ok: apply bypassed {pruned}, restore reverted")


def test_protection():
    # even if layer 0 and last are most redundant, they must never be pruned
    deltas = [0.001, 1.0, 1.0, 1.0, 0.002]  # layer0 and layer4(last) most redundant
    lm = FakeLM(deltas)
    p = DepthPruner(lm)
    p.install_calibration_hooks()
    lm.model(torch.randn(1, 5, 8))
    p.finalize_calibration()
    pruned = p.apply(count=2, min_layer=2)
    assert 0 not in pruned and 4 not in pruned, f"protected layers leaked into {pruned}"
    print(f"ok: protection held; pruned={pruned} (0 and last excluded)")


if __name__ == "__main__":
    test_bypass_is_identity()
    test_ranking_orders_by_redundancy()
    test_apply_and_restore()
    test_protection()
    print("\nALL PASS")
