import numpy as np
import pytest

from trajectory_explorer.arch import TensorKey, check_compatible, classify, is_buffer
from trajectory_explorer.errors import ArchitectureMismatch
from trajectory_explorer.reader import Checkpoint


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # GPT-NeoX / Pythia
        ("gpt_neox.embed_in.weight", TensorKey(None, "embed")),
        ("embed_out.weight", TensorKey(None, "unembed")),
        ("gpt_neox.final_layer_norm.bias", TensorKey(None, "norm")),
        ("gpt_neox.layers.0.attention.query_key_value.weight", TensorKey(0, "attn_qkv")),
        ("gpt_neox.layers.3.attention.query_key_value.bias", TensorKey(3, "attn_qkv")),
        ("gpt_neox.layers.5.attention.dense.weight", TensorKey(5, "attn_out")),
        ("gpt_neox.layers.1.mlp.dense_h_to_4h.weight", TensorKey(1, "mlp_in")),
        ("gpt_neox.layers.1.mlp.dense_4h_to_h.bias", TensorKey(1, "mlp_out")),
        ("gpt_neox.layers.2.post_attention_layernorm.weight", TensorKey(2, "norm")),
        # Llama-style / SmolLM2
        ("model.embed_tokens.weight", TensorKey(None, "embed")),
        ("lm_head.weight", TensorKey(None, "unembed")),
        ("model.norm.weight", TensorKey(None, "norm")),
        ("model.layers.29.self_attn.k_proj.weight", TensorKey(29, "attn_qkv")),
        ("model.layers.0.self_attn.o_proj.weight", TensorKey(0, "attn_out")),
        ("model.layers.4.mlp.gate_proj.weight", TensorKey(4, "mlp_in")),
        ("model.layers.4.mlp.up_proj.weight", TensorKey(4, "mlp_in")),
        ("model.layers.4.mlp.down_proj.weight", TensorKey(4, "mlp_out")),
        ("model.layers.7.input_layernorm.weight", TensorKey(7, "norm")),
        # Unknown architecture: generic fallback
        ("transformer.h.2.ln_1.weight", TensorKey(2, "norm")),
        ("transformer.h.2.attn.c_attn.weight", TensorKey(2, "other")),
    ],
)
def test_classify_neox_llama_and_fallback_names(name, expected):
    assert classify(name) == expected


@pytest.mark.unit
def test_buffers_are_filtered_by_name():
    assert is_buffer("gpt_neox.layers.0.attention.bias")
    assert is_buffer("gpt_neox.layers.0.attention.masked_bias")
    assert is_buffer("gpt_neox.layers.0.attention.rotary_emb.inv_freq")
    # Real parameters whose names also end in "bias" are not buffers.
    assert not is_buffer("gpt_neox.layers.0.attention.query_key_value.bias")
    assert not is_buffer("gpt_neox.layers.0.attention.dense.bias")


def _pythia_like(dtype: type, *, with_buffers: bool, hidden: int = 4) -> dict:
    t = {
        "gpt_neox.embed_in.weight": np.zeros((10, hidden), dtype),
        "gpt_neox.layers.0.attention.query_key_value.weight": np.zeros((3 * hidden, hidden), dtype),
        "gpt_neox.layers.0.attention.query_key_value.bias": np.zeros((3 * hidden,), dtype),
    }
    if with_buffers:  # what pythia `main` has on top of the step* revisions
        t["gpt_neox.layers.0.attention.bias"] = np.ones((1, 1, 4, 4), np.uint8)
        t["gpt_neox.layers.0.attention.masked_bias"] = np.array(-1e4, np.float16)
        t["gpt_neox.layers.0.attention.rotary_emb.inv_freq"] = np.ones((2,), np.float16)
    return t


@pytest.mark.unit
def test_compatibility_ignores_buffers_and_dtype_but_reports_mismatches(make_checkpoint):
    main_like = make_checkpoint("main.safetensors", _pythia_like(np.float16, with_buffers=True))
    step_like = make_checkpoint("step.safetensors", _pythia_like(np.float32, with_buffers=False))
    with Checkpoint(main_like) as a, Checkpoint(step_like) as b:
        shared = check_compatible(a.specs(), b.specs())
    assert len(shared) == 3

    other = _pythia_like(np.float32, with_buffers=False, hidden=6)
    other["gpt_neox.layers.1.mlp.dense_4h_to_h.weight"] = np.zeros((6, 24), np.float32)
    del other["gpt_neox.layers.0.attention.query_key_value.bias"]
    mismatched = make_checkpoint("other.safetensors", other)
    with (
        Checkpoint(step_like) as a,
        Checkpoint(mismatched) as b,
        pytest.raises(ArchitectureMismatch) as exc,
    ):
        check_compatible(a.specs(), b.specs())

    message = str(exc.value)
    assert exc.value.exit_code == 3
    assert "Only in A (1):\n  - gpt_neox.layers.0.attention.query_key_value.bias" in message
    assert "Only in B (1):\n  - gpt_neox.layers.1.mlp.dense_4h_to_h.weight" in message
    assert "gpt_neox.embed_in.weight: A[10, 4] vs B[10, 6]" in message
