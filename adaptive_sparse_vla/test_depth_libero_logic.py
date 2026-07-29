"""Depth-pruning logic against a REAL (tiny, randomly-initialised) vendored
Emu3MoE -- no checkpoint, no GPU.

The logic under test lives in `depth_prune.py`, shared by every LIBERO
backbone. That sharing is the point: the cross-backbone claim is that
exploitable depth redundancy differs by backbone (Emu3 absorbs 8 bypassed
layers where Gemma2 broke at 1), which only means something if the selection
rule is identical everywhere. So this file tests the shared module, not a copy.

What has to hold, in order of how easy it is to get wrong:

  1. BypassDecoderLayer plugs into a real decoder loop (call signature and
     return-tuple indexing) and multi-step generate() still runs -- this is the
     thing that silently corrupts the KV cache if wrong.
  2. apply/restore is an exact round-trip *across a state change*, so a
     per-episode reset really restores the full model before re-ranking.
  3. The ranking honours --depth-min-layer and the adjacency gap, and deep is a
     strict PREFIX of shallow, so the controller's switch only ever adds layers
     rather than swapping to a different set.
  4. Bypassing changes the output (it is not a silent no-op).
  5. Both redundancy measurements -- direct forward (Emu3) and forward hooks
     (OpenVLA, whose generate is wrapped in predict_action) -- agree, which is
     what lets the two backbones be compared at all.
  6. Layer discovery finds the stack through OpenVLA's extra wrapper level.

Run it on CPU with the transformers pin the eval uses (4.44.2 for UniVLA):

    cd adaptive_sparse_vla && python test_depth_libero_logic.py
    # -> ALL 27 DEPTH-PRUNING CHECKS PASSED
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("UNIVLA_ROOT", os.path.join(os.path.dirname(_HERE), "UniVLA"))
sys.path[:0] = [_HERE, ROOT, os.path.join(ROOT, "reference", "Emu3")]

import transformers  # noqa: E402
from emu3.mllm.configuration_emu3 import Emu3MoEConfig  # noqa: E402
import emu3.mllm.modeling_emu3 as M  # noqa: E402
from transformers import GenerationConfig  # noqa: E402

from depth_prune import (  # noqa: E402
    DepthPruner,
    find_decoder_layers,
    measure_redundancy_with_hooks,
)

print("transformers", transformers.__version__)

N_LAYERS = 8
cfg = Emu3MoEConfig(
    vocab_size=512, hidden_size=64, intermediate_size=128,
    num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=4096, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    img_token_id=3, boi_token_id=4, eoi_token_id=5, eol_token_id=6, eof_token_id=7,
    boa_token_id=8, eov_token_id=9, bov_token_id=10, action_experts=False,
    attention_dropout=0.0,
)
cfg._attn_implementation = "sdpa"
torch.manual_seed(0)
model = M.Emu3MoE(cfg).eval()

p = DepthPruner(model, ctrl=True, deep=2, shallow=4, close_steps=2,
                min_layer=0.5, min_gap=1)

ids = torch.randint(11, 512, (1, 96))
mask = torch.ones_like(ids)
ok = 0
# Captured before anything is bypassed, so identity comparisons later are
# against the genuine modules rather than whatever a previous step installed.
ORIGINALS = {i: model.model.layers[i] for i in range(N_LAYERS)}


def check(label, cond, extra=""):
    global ok
    assert cond, f"FAIL: {label} {extra}"
    ok += 1
    print(f"  ok  {label} {extra}")


def direct_redundancy():
    with torch.inference_mode():
        hs = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                         output_hidden_states=True, return_dict=True).hidden_states
    return [
        float(1.0 - torch.nn.functional.cosine_similarity(
            a.float(), b.float(), dim=-1).mean().item())
        for a, b in zip(hs[:-1], hs[1:])
    ]


print("\n[0] layer discovery")
check("finds the Emu3 stack", find_decoder_layers(model) is model.model.layers)
check("pruner sees the right layer count", p.n_layers() == N_LAYERS)


class _Wrapped(torch.nn.Module):
    """OpenVLA shape: the decoder sits one wrapper deeper, at
    model.language_model.model.layers."""

    def __init__(self, inner):
        super().__init__()
        self.language_model = inner


check("finds the stack through OpenVLA's extra wrapper level",
      find_decoder_layers(_Wrapped(model)) is model.model.layers)

print("\n[1] redundancy calibration")
imp = direct_redundancy()
check("one score per layer", imp is not None and len(imp) == N_LAYERS, f"n={len(imp)}")
check("scores finite", all(v == v and abs(v) < 1e6 for v in imp))

print("\n[2] the two measurement paths agree")
hooked = measure_redundancy_with_hooks(
    model.model.layers,
    lambda: model(input_ids=ids, attention_mask=mask, use_cache=False),
)
check("hooks return one score per layer",
      hooked is not None and len(hooked) == N_LAYERS)
check("hook scores match the direct forward",
      all(abs(a - b) < 1e-4 for a, b in zip(imp, hooked)),
      f"maxdiff={max(abs(a - b) for a, b in zip(imp, hooked)):.2e}")
check("both paths rank layers identically",
      p.rank_layers(imp) == p.rank_layers(hooked),
      f"{p.rank_layers(imp)}")

print("\n[3] ranking respects --depth-min-layer and the adjacency gap")
ranking = p.rank_layers(imp)
check("ranking covers only eligible layers", set(ranking) == set(range(4, 8)),
      f"ranking={ranking}")
check("no two adjacent layers in the gap-respecting prefix",
      abs(ranking[0] - ranking[1]) > 1, f"prefix={ranking[:2]}")
check("most-redundant-first", imp[ranking[0]] <= imp[ranking[1]],
      f"{[imp[i] for i in ranking[:2]]}")

print("\n[4] deep is a strict prefix of shallow (nested states)")
p.calibrate(imp)
deep_set = set(p._active)
check("deep bypasses depth_deep layers", len(deep_set) == 2, f"{sorted(deep_set)}")
p._state = "shallow"
p._apply_state()
shallow_set = set(p._active)
check("shallow bypasses depth_shallow layers", len(shallow_set) == 4, f"{sorted(shallow_set)}")
check("deep subset of shallow", deep_set < shallow_set,
      f"deep={sorted(deep_set)} shallow={sorted(shallow_set)}")

print("\n[5] bypass/restore is an exact round-trip")
# Entering from the shallow state: apply() must restore what is already
# bypassed before installing a different set, or bypasses accumulate and can
# never be undone.
p.apply([5, 7])
swapped = [i for i in range(N_LAYERS)
           if type(model.model.layers[i]).__name__ == "BypassDecoderLayer"]
check("exactly the requested layers are bypassed", swapped == [5, 7], f"{swapped}")
p.restore()
check("every slot holds the original module object again",
      all(model.model.layers[i] is ORIGINALS[i] for i in range(N_LAYERS)),
      "(incl. layers left bypassed by the previous state)")
check("bookkeeping cleared", p._active == () and not p._originals)

print("\n[6] the grasp signal drives a one-way deep -> shallow switch")
q = DepthPruner(model, ctrl=True, deep=2, shallow=4, close_steps=2, min_layer=0.5)
q.calibrate(imp)
q.note_gripper(False)
check("an open gripper does not switch", q._state == "deep")
q.note_gripper(True)
check("one close is below the hysteresis", q._state == "deep",
      f"count={q.close_gripper_num}")
q.note_gripper(True)
check("two consecutive closes switch to shallow", q._state == "shallow")
q.note_gripper(False)
check("the switch is one-way", q._state == "shallow")
check("switches are counted for the summary",
      q.summary()["episodes_reaching_shallow"] == 1)
q.restore()

print("\n[7] generate() survives a bypassed stack (KV cache stays contiguous)")
GEN = GenerationConfig(pad_token_id=0, bos_token_id=1, eos_token_id=2, do_sample=False)
short = torch.randint(11, 512, (1, 32))
with torch.no_grad():
    base_out = model.generate(short, GEN, max_new_tokens=12,
                              attention_mask=torch.ones_like(short))
check("unpruned generate produces 12 new tokens",
      base_out.shape[1] == short.shape[1] + 12, f"shape={tuple(base_out.shape)}")

p.apply([5, 7])
with torch.no_grad():
    pruned_out = model.generate(short, GEN, max_new_tokens=12,
                                attention_mask=torch.ones_like(short))
check("pruned generate produces 12 new tokens",
      pruned_out.shape[1] == short.shape[1] + 12, f"shape={tuple(pruned_out.shape)}")

print("\n[8] bypassing is not a silent no-op")
with torch.no_grad():
    lg_pruned = model(input_ids=short, attention_mask=torch.ones_like(short)).logits
p.restore()
with torch.no_grad():
    lg_full = model(input_ids=short, attention_mask=torch.ones_like(short)).logits
check("logits change when layers are bypassed",
      not torch.allclose(lg_full, lg_pruned, atol=1e-4),
      f"maxdiff={(lg_full - lg_pruned).abs().max().item():.4f}")
check("no NaNs introduced", not torch.isnan(lg_pruned).any())

print("\n[9] cached generate matches uncached greedy decode WHILE pruned")
p.apply([5, 7])
with torch.no_grad():
    cached = model.generate(short, GEN, max_new_tokens=8,
                            attention_mask=torch.ones_like(short))
    manual = short.clone()
    for _ in range(8):
        nxt = model(input_ids=manual, attention_mask=torch.ones_like(manual),
                    use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        manual = torch.cat([manual, nxt], dim=1)
check("cached == uncached under pruning", torch.equal(cached, manual),
      f"cached={cached[0, -8:].tolist()} manual={manual[0, -8:].tolist()}")
p.restore()

print(f"\nALL {ok} DEPTH-PRUNING CHECKS PASSED")
