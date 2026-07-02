"""
Training-free decoder-layer bypass, used by both static/depth-controller
pruning (`inference.py`) and self-speculative decoding
(`emu3_self_spec_decode.py`) as the "draft" (few layers bypassed) mode.

Split into its own lightweight module (torch only, no `transforms3d`/
`simpler_env`/etc.) so it can be imported by CPU unit tests without pulling
in the full inference stack's heavy dependencies.
"""
from __future__ import annotations

from typing import Any, Optional

import torch


class BypassDecoderLayer(torch.nn.Module):
    """
    The layer's compute is skipped (hidden states pass through unchanged), but it
    still writes a correctly-shaped placeholder KV into the cache. This keeps the
    per-layer cache contiguous: transformers' DynamicCache (>=4.51) indexes by
    layer_idx, so a layer that never calls `update` leaves a gap and a later layer's
    `update` hits `IndexError: list index out of range`. The placeholder is never
    read (this layer has no attention), so zeros are safe.
    """

    def __init__(self, layer_idx: int, num_kv_heads: Optional[int] = None,
                 head_dim: Optional[int] = None):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        if (
            use_cache
            and past_key_value is not None
            and hasattr(past_key_value, "update")
            and self.num_kv_heads is not None
            and self.head_dim is not None
        ):
            bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            dummy = torch.zeros(
                bsz, self.num_kv_heads, seq_len, self.head_dim,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            past_key_value.update(dummy, dummy, self.layer_idx, {})
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs
