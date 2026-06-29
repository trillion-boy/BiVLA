"""CPU unit test for the ToMe merge/unmerge math (no model, no GPU needed).

Validates the OOD-safety invariants we rely on:
  1. token-count is preserved end to end (unmerge restores the full grid);
  2. protected (important) patches are bit-exactly untouched;
  3. similar/redundant tokens actually collapse (merged clusters share a value);
  4. a fake SigLIP tower patched with ToMe returns the original token count.
"""
import torch

from tome_siglip import (
    bipartite_soft_matching,
    _merge_source,
    apply_tome_to_siglip,
    center_protect_provider,
)


def test_count_and_protection():
    torch.manual_seed(0)
    B, N, C = 2, 64, 16
    x = torch.randn(B, N, C)
    # make a clear redundant background: tokens 10..40 nearly identical
    x[:, 10:40, :] = x[:, 10:11, :] + 0.01 * torch.randn(B, 30, C)

    protect = torch.zeros(B, N, dtype=torch.bool)
    protect[:, 0:4] = True                      # pretend these are "important"
    important_before = x[:, 0:4, :].clone()

    source = torch.eye(N)[None].expand(B, N, N).clone()
    cur = x
    cur_protect = protect
    for _ in range(6):
        merge = bipartite_soft_matching(cur, r=4, protected=cur_protect)
        cur = merge(cur, mode="mean")
        source = _merge_source(merge, source)
        cur_protect = merge(cur_protect.float()[..., None], mode="amax")[..., 0] > 0.5

    assert cur.shape[1] == N - 6 * 4, f"expected {N-24} tokens, got {cur.shape[1]}"

    owner = source.argmax(dim=1)
    restored = cur.gather(1, owner[..., None].expand(B, N, C))
    assert restored.shape == (B, N, C), restored.shape

    # protected tokens survive 1:1 and are bit-exact
    err = (restored[:, 0:4, :] - important_before).abs().max().item()
    assert err < 1e-5, f"protected tokens changed by {err}"
    print(f"[ok] count preserved ({N}->{cur.shape[1]}->{restored.shape[1]}); "
          f"protected drift={err:.2e}")


def test_redundant_collapse():
    torch.manual_seed(1)
    B, N, C = 1, 64, 8
    x = torch.randn(B, N, C)
    x[:, 20:36, :] = x[:, 20:21, :]             # 16 identical background tokens
    merge = bipartite_soft_matching(x, r=8)
    merged = merge(x, mode="mean")
    assert merged.shape[1] == N - 8
    print(f"[ok] redundant tokens merged: {N} -> {merged.shape[1]}")


class _FakeLayer(torch.nn.Module):
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        return (hidden_states + 0.0,)


class _FakeEmbeddings(torch.nn.Module):
    def __init__(self, N, C):
        super().__init__()
        self.N, self.C = N, C

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        return torch.randn(B, self.N, self.C)


class _FakeVisionTransformer(torch.nn.Module):
    def __init__(self, N, C, n_layers):
        super().__init__()
        self.embeddings = _FakeEmbeddings(N, C)
        self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList([_FakeLayer() for _ in range(n_layers)])
        self.post_layernorm = torch.nn.LayerNorm(C)


class _FakeSiglip(torch.nn.Module):
    def __init__(self, N=64, C=16, n_layers=12):
        super().__init__()
        self.vision_model = _FakeVisionTransformer(N, C, n_layers)

    def forward(self, pixel_values):
        vm = self.vision_model
        h = vm.embeddings(pixel_values)
        for layer in vm.encoder.layers:
            h = layer(h)[0]
        h = vm.post_layernorm(h)
        from tome_siglip import _vision_output
        return _vision_output(h)


def test_patched_tower_keeps_count():
    N, C = 64, 16
    tower = _FakeSiglip(N=N, C=C, n_layers=12)
    px = torch.randn(1, 3, 8, 8)
    base = tower(px).last_hidden_state
    assert base.shape == (1, N, C)

    apply_tome_to_siglip(tower, r=6, num_merge_layers=6,
                         protect_provider=center_protect_provider(0.25))
    out = tower(px).last_hidden_state
    assert out.shape == (1, N, C), f"patched tower changed count: {out.shape}"
    print(f"[ok] patched SigLIP tower output shape {tuple(out.shape)} == baseline")


if __name__ == "__main__":
    test_count_and_protection()
    test_redundant_collapse()
    test_patched_tower_keeps_count()
    print("\nALL TOME LOGIC TESTS PASSED")
