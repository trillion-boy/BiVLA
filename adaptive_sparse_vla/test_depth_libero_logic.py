"""Depth-pruning logic on the LIBERO UniVLA wrapper, against a REAL (tiny,
randomly-initialised) vendored Emu3MoE -- no checkpoint, no GPU.

What this has to prove, in order of how easy it is to get wrong:
  1. BypassDecoderLayer actually plugs into the vendored Emu3 decoder loop
     (call signature + return-tuple indexing) and multi-step generate() still
     runs -- this is the thing that silently corrupts the KV cache if wrong.
  2. _apply_pruning / _restore_layers is an exact round-trip (same module
     objects back in the same slots), so a per-episode reset really does
     restore the full model before re-calibrating.
  3. The ranking respects --depth-min-layer and the adjacency gap, and deep is
     a strict PREFIX of shallow (nested), so the controller's state switch only
     ever adds layers rather than swapping to a different set.
  4. Bypassing changes the output (it is not a silent no-op).

Run it on CPU with the same transformers pin the eval uses (4.44.2):

    cd adaptive_sparse_vla && python test_depth_libero_logic.py
    # -> ALL 16 DEPTH-PRUNING CHECKS PASSED
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

from inference_libero import EmuVLALiberoInference  # noqa: E402

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

# Build the wrapper without touching __init__ (which would load a checkpoint,
# a tokenizer and a vision tokenizer). Only the depth-axis state is needed.
p = object.__new__(EmuVLALiberoInference)
p.model = model
p.device = "cpu"
p.depth_prune = 0
p.depth_ctrl = True
p.depth_deep = 2
p.depth_shallow = 4
p.depth_close_steps = 2
p.depth_min_layer = 0.5
p.depth_min_gap = 1
p._original_decoder_layers = {}
p._active_prune_layers = ()
p._depth_ranking = []
p._depth_ranking_ready = False
p._depth_state = "deep"
p._depth_calibrated = False
p.close_gripper_num = 0

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


print("\n[1] redundancy calibration")
imp = p._layer_redundancy(ids, mask)
check("one score per layer", imp is not None and len(imp) == N_LAYERS, f"n={len(imp)}")
check("scores finite", all(v == v and abs(v) < 1e6 for v in imp))

print("\n[2] ranking respects --depth-min-layer and the adjacency gap")
ranking = p._rank_layers(imp)
check("ranking covers only eligible layers", set(ranking) == set(range(4, 8)),
      f"ranking={ranking}")
gap_prefix = ranking[:2]
check("no two adjacent layers in the gap-respecting prefix",
      abs(gap_prefix[0] - gap_prefix[1]) > 1, f"prefix={gap_prefix}")
scores_in_order = [imp[i] for i in ranking[:2]]
check("most-redundant-first", scores_in_order[0] <= scores_in_order[1],
      f"{scores_in_order}")

print("\n[3] deep is a strict prefix of shallow (nested states)")
p._depth_ranking = ranking
p._depth_ranking_ready = True
p._depth_state = "deep"
p._depth_apply_state()
deep_set = set(p._active_prune_layers)
check("deep bypasses depth_deep layers", len(deep_set) == 2, f"{sorted(deep_set)}")
p._depth_state = "shallow"
p._depth_apply_state()
shallow_set = set(p._active_prune_layers)
check("shallow bypasses depth_shallow layers", len(shallow_set) == 4, f"{sorted(shallow_set)}")
check("deep subset of shallow", deep_set < shallow_set,
      f"deep={sorted(deep_set)} shallow={sorted(shallow_set)}")

print("\n[4] bypass/restore is an exact round-trip")
# Entering from step 3's shallow state ([4,5,6,7] bypassed): _apply_pruning must
# restore what is already bypassed before installing a different set, or the
# stack accumulates BypassDecoderLayers it can never undo.
p._apply_pruning([5, 7])
swapped = [i for i in range(N_LAYERS)
           if type(model.model.layers[i]).__name__ == "BypassDecoderLayer"]
check("exactly the requested layers are bypassed", swapped == [5, 7], f"{swapped}")
p._restore_layers()
check("every slot holds the original module object again",
      all(model.model.layers[i] is ORIGINALS[i] for i in range(N_LAYERS)),
      "(incl. layers left bypassed by the previous state)")
check("bookkeeping cleared", p._active_prune_layers == () and not p._original_decoder_layers)

print("\n[5] generate() survives a bypassed stack (KV cache stays contiguous)")
GEN = GenerationConfig(pad_token_id=0, bos_token_id=1, eos_token_id=2, do_sample=False)
short = torch.randint(11, 512, (1, 32))
with torch.no_grad():
    base_out = model.generate(short, GEN, max_new_tokens=12,
                              attention_mask=torch.ones_like(short))
check("unpruned generate produces 12 new tokens",
      base_out.shape[1] == short.shape[1] + 12, f"shape={tuple(base_out.shape)}")

p._apply_pruning([5, 7])
with torch.no_grad():
    pruned_out = model.generate(short, GEN, max_new_tokens=12,
                                attention_mask=torch.ones_like(short))
check("pruned generate produces 12 new tokens",
      pruned_out.shape[1] == short.shape[1] + 12, f"shape={tuple(pruned_out.shape)}")

print("\n[6] bypassing is not a silent no-op")
with torch.no_grad():
    lg_pruned = model(input_ids=short, attention_mask=torch.ones_like(short)).logits
p._restore_layers()
with torch.no_grad():
    lg_full = model(input_ids=short, attention_mask=torch.ones_like(short)).logits
check("logits change when layers are bypassed",
      not torch.allclose(lg_full, lg_pruned, atol=1e-4),
      f"maxdiff={(lg_full - lg_pruned).abs().max().item():.4f}")
check("no NaNs introduced", not torch.isnan(lg_pruned).any())

print("\n[7] cached generate matches uncached greedy decode WHILE pruned")
p._apply_pruning([5, 7])
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
p._restore_layers()

print(f"\nALL {ok} DEPTH-PRUNING CHECKS PASSED")
