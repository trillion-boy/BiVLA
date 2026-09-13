

## Summary

| Model | Attention backend in the current harness | Important qualification |
| --- | --- | --- |
| **SpatialVLA** | **Mixed/native configuration**; the custom Gemma2 text decoder falls back to **eager attention** by default. | The model contains multiple components, including the language decoder, SigLIP vision encoder, and ZoeDepth module, so there is no single attention backend for the entire model. The default `sdpa` runner option does not force all components to use SDPA. |
| **CogACT** | **Implicit SDPA** for the Llama language decoder under the current `transformers==4.47.0` environment. | This is selected implicitly during model construction rather than explicitly requested by the harness and is therefore version-dependent. The DiT action head uses a separate attention implementation. |
| **CronusVLA** | **Implicit SDPA** for the Qwen2.5 language decoder under `transformers==4.47.0`. | The backend is selected implicitly and is version-dependent. The action decoder independently uses timm's fused-attention path when available, otherwise standard matrix-multiplication/softmax attention. |
| **MiniVLA** | **Implicit, version-dependent attention**, currently resolving to SDPA under `transformers==4.47.0`. | MiniVLA itself pins `transformers==4.40.1`, while the current evaluation environment uses 4.47.0. Therefore it should not be described as explicitly using SDPA or FlashAttention-2. |
| **SmolVLA** | **Explicit PyTorch SDPA API**. | The attention implementation explicitly calls PyTorch SDPA, but the underlying CUDA kernel is still chosen dynamically by PyTorch. It therefore does not guarantee FlashAttention-2. |

## SpatialVLA

SpatialVLA should not be labeled simply as an SDPA model.

Its evaluation configuration is effectively **mixed/native** because SpatialVLA consists of several nested components with potentially different attention implementations. Most importantly, its custom Gemma2 language decoder explicitly falls back from the default SDPA configuration to **eager attention** because of Gemma2-specific logits-softcapping behavior.

Therefore, for the default SpatialVLA evaluation:

**Text decoder:** eager attention  
**Overall model:** mixed/native attention configuration

The vision encoder, depth encoder, and language decoder should be treated as separate components if exact backend information is required.

An `eager` override can explicitly request eager attention for compatible modules, but the nominal `sdpa` setting should not be interpreted as forcing SDPA throughout SpatialVLA.

## CogACT

CogACT does not explicitly request an attention implementation from the evaluation harness.

Its Llama language decoder is constructed using the model configuration, allowing Transformers to choose the appropriate attention implementation. Under the current environment with `transformers==4.47.0`, the decoder resolves to **SDPA**.

Thus the appropriate description is:

**Llama decoder:** implicit SDPA under the current Transformers version

This is a runtime-dependent result rather than a property guaranteed by the CogACT evaluation harness.

CogACT also contains a DiT-based action head whose attention implementation is separate from the Hugging Face Llama decoder. Consequently, the entire CogACT architecture should not be described as uniformly using Hugging Face SDPA.

## CronusVLA

CronusVLA follows the same general pattern as CogACT.

Its Qwen2.5 language decoder does not receive an explicit attention-backend request. Instead, the backend is resolved automatically by Transformers during model construction.

Under the current `transformers==4.47.0` environment, the Qwen2.5 decoder resolves to **SDPA**.

Thus the appropriate description is:

**Qwen2.5 decoder:** implicit SDPA under the current Transformers version

CronusVLA's action decoder has its own attention implementation. When timm enables fused attention, it uses PyTorch SDPA; otherwise, it falls back to explicit query-key multiplication, softmax, and value aggregation.

Therefore, as with CogACT, attention should be described separately for the language and action components.

## MiniVLA

MiniVLA also relies on automatic attention selection rather than explicitly requesting SDPA or FlashAttention-2.

Under the current evaluation environment with `transformers==4.47.0`, its Qwen2.5 decoder resolves to **SDPA**.

However, the MiniVLA source environment pins `transformers==4.40.1`. Because attention implementation selection can differ across Transformers versions, MiniVLA should be described as:

**Implicit/version-dependent attention, currently observed as SDPA**

It should not be reported as an explicitly configured SDPA or FlashAttention-2 model.

For reproducibility, the Transformers version used during evaluation is particularly important for MiniVLA.

## SmolVLA

SmolVLA is the most explicit case.

The local evaluation implementation directly uses PyTorch's:

`torch.nn.functional.scaled_dot_product_attention`

Therefore its backend can accurately be described as:

**Explicit PyTorch SDPA API**

However, this still does not guarantee FlashAttention-2. PyTorch dynamically chooses the underlying SDPA kernel based on the runtime environment and input characteristics.

The upstream SmolVLM configuration may mention FlashAttention-2, but that is not the attention path used by the local SimpleWidowX evaluation implementation.

## Reproducibility Implications

The five models should **not** be grouped together under a single "SDPA" label.

A more accurate categorization is:

- **SpatialVLA:** mixed/native; Gemma2 language decoder uses eager attention.
- **CogACT:** implicitly resolves to SDPA in the current Transformers environment.
- **CronusVLA:** implicitly resolves to SDPA for the language decoder; action-decoder attention is handled separately.
- **MiniVLA:** implicit and version-dependent; currently resolves to SDPA.
- **SmolVLA:** explicitly uses the PyTorch SDPA API.

For reproducible latency and efficiency comparisons, evaluations should record at minimum:

- PyTorch version;
- Transformers version;
- configured or resolved attention implementation;
- attention module class used by the language backbone;
- separate attention implementations for auxiliary or action-decoder components where applicable.

For SmolVLA, it is also useful to record that the local evaluation implementation explicitly replaces the joint model's attention path with PyTorch SDPA.

For SpatialVLA, attention should be reported **per major component** rather than assigning a single backend to the complete architecture.

The most important distinction for reporting is therefore **explicit SDPA vs. implicit SDPA vs. eager/mixed attention**, rather than treating every SDPA-related execution path as FlashAttention-2.