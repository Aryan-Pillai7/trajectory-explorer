"""Map tensor names to (layer, component), filter out buffers, and check compatibility.

Components (heatmap columns): embed, attn_qkv, attn_out, mlp_in, mlp_out, norm, unembed, other.
Rules cover GPT-NeoX (Pythia) and Llama-style (SmolLM2) names; anything else falls back to a
generic layer-number + keyword guess, or "other".
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from trajectory_explorer.errors import ArchitectureMismatch
from trajectory_explorer.reader import TensorSpec

COMPONENTS: tuple[str, ...] = (
    "embed", "attn_qkv", "attn_out", "mlp_in", "mlp_out", "norm", "unembed", "other",
)  # fmt: skip

# Non-parameter buffers stored in some checkpoints. Measured in pythia-70m `main` (absent from
# the step* revisions): attention.bias (U8 causal mask), attention.masked_bias (F16 scalar) and
# attention.rotary_emb.inv_freq (F16). inv_freq is a float tensor, so dtype filtering alone would
# miss it; buffers are filtered by name.
_BUFFER_RE = re.compile(r"(?:^|\.)attention\.(?:bias|masked_bias)$|(?:^|\.)rotary_emb\.inv_freq$")

_L = r"(?P<layer>\d+)"
_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), component)
    for pattern, component in (
        # GPT-NeoX (Pythia)
        (r"^gpt_neox\.embed_in\.weight$", "embed"),
        (r"^embed_out\.weight$", "unembed"),
        (r"^gpt_neox\.final_layer_norm\.(?:weight|bias)$", "norm"),
        (rf"^gpt_neox\.layers\.{_L}\.attention\.query_key_value\.(?:weight|bias)$", "attn_qkv"),
        (rf"^gpt_neox\.layers\.{_L}\.attention\.dense\.(?:weight|bias)$", "attn_out"),
        (rf"^gpt_neox\.layers\.{_L}\.mlp\.dense_h_to_4h\.(?:weight|bias)$", "mlp_in"),
        (rf"^gpt_neox\.layers\.{_L}\.mlp\.dense_4h_to_h\.(?:weight|bias)$", "mlp_out"),
        (
            rf"^gpt_neox\.layers\.{_L}\.(?:input_layernorm|post_attention_layernorm)"
            r"\.(?:weight|bias)$",
            "norm",
        ),
        # Llama-style (SmolLM2, Llama, Mistral, Qwen2)
        (r"^model\.embed_tokens\.weight$", "embed"),
        (r"^lm_head\.weight$", "unembed"),
        (r"^model\.norm\.weight$", "norm"),
        (rf"^model\.layers\.{_L}\.self_attn\.[qkv]_proj\.(?:weight|bias)$", "attn_qkv"),
        (rf"^model\.layers\.{_L}\.self_attn\.o_proj\.(?:weight|bias)$", "attn_out"),
        (rf"^model\.layers\.{_L}\.mlp\.(?:gate|up)_proj\.weight$", "mlp_in"),
        (rf"^model\.layers\.{_L}\.mlp\.down_proj\.weight$", "mlp_out"),
        (rf"^model\.layers\.{_L}\.(?:input_layernorm|post_attention_layernorm)\.weight$", "norm"),
    )
)
_GENERIC_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|layer|h|blocks|block)\.(\d+)\.")


@dataclass(frozen=True)
class TensorKey:
    """Where a tensor sits in the model: layer index (None = outside the layer stack)."""

    layer: int | None
    component: str


def is_buffer(name: str) -> bool:
    """True for known non-parameter buffers (masks, rotary tables)."""
    return _BUFFER_RE.search(name) is not None


def classify(name: str) -> TensorKey:
    """Map a tensor name to its layer and component."""
    for pattern, component in _RULES:
        match = pattern.match(name)
        if match:
            layer = match.groupdict().get("layer")
            return TensorKey(int(layer) if layer is not None else None, component)

    generic = _GENERIC_LAYER_RE.search(name)
    layer = int(generic.group(1)) if generic else None
    lowered = name.lower()
    if "norm" in lowered or re.search(r"(?:^|\.)ln_?\w*\.", lowered):
        component = "norm"
    elif "embed" in lowered or "wte" in lowered:
        component = "embed"
    else:
        component = "other"
    return TensorKey(layer, component)


def parameter_specs(specs: Mapping[str, TensorSpec]) -> tuple[dict[str, TensorSpec], list[str]]:
    """Split specs into (parameters, skipped names). Skips buffers and non-float tensors."""
    params: dict[str, TensorSpec] = {}
    skipped: list[str] = []
    for name, spec in specs.items():
        if is_buffer(name) or not spec.is_float:
            skipped.append(name)
        else:
            params[name] = spec
    return params, skipped


def check_compatible(
    specs_a: Mapping[str, TensorSpec], specs_b: Mapping[str, TensorSpec]
) -> list[str]:
    """Return the shared parameter names (in A's order) or raise ArchitectureMismatch.

    Buffers and non-float tensors are ignored. Dtypes may differ (e.g. F16 vs F32); names and
    shapes must match exactly.
    """
    params_a, _ = parameter_specs(specs_a)
    params_b, _ = parameter_specs(specs_b)
    only_a = [n for n in params_a if n not in params_b]
    only_b = [n for n in params_b if n not in params_a]
    shape_mismatches = [
        (n, params_a[n].shape, params_b[n].shape)
        for n in params_a
        if n in params_b and params_a[n].shape != params_b[n].shape
    ]
    if only_a or only_b or shape_mismatches:
        raise ArchitectureMismatch(only_a, only_b, shape_mismatches)
    return list(params_a)
