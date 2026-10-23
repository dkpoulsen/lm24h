import torch, pytest
from lm24h.model import ModelConfig, Transformer
from lm24h.export_hf import hf_config, convert_state

def tiny(): return ModelConfig(vocab_size=128, n_layer=2, d_model=64, n_head=4, n_kv_head=2, head_dim=16, ffn_dim=128, max_seq_len=32)

def test_shapes_and_param_count():
    c = tiny(); m = Transformer(c); assert sum(p.numel() for p in m.parameters()) == c.n_params()
    assert m(torch.randint(0, 128, (2, 16))).shape == (2, 16, 128)

def test_causality():
    m = Transformer(tiny()).eval(); x = torch.randint(0, 128, (1, 16)); y = x.clone(); y[0, 10:] = 5
    with torch.no_grad(): a, b = m(x), m(y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)

def test_overfits_tiny_batch():
    torch.manual_seed(0); m = Transformer(tiny()); x = torch.randint(0, 128, (4, 17)); opt = torch.optim.AdamW(m.parameters(), 3e-3)
    l0 = None
    for _ in range(60):
        l = m(x[:, :-1], x[:, 1:]); l0 = l0 or l.item(); l.backward(); opt.step(); opt.zero_grad()
    assert l.item() < l0 * 0.5

def test_hf_qwen3_parity(tmp_path):
    transformers = pytest.importorskip("transformers"); from transformers import Qwen3Config, Qwen3ForCausalLM
    from safetensors.torch import load_file
    c = tiny(); m = Transformer(c).eval()
    hc = Qwen3Config(**{k: v for k, v in hf_config(c).items() if k not in ("architectures", "model_type", "torch_dtype")})
    hf = Qwen3ForCausalLM(hc).eval()
    missing, unexpected = hf.load_state_dict(convert_state(m.state_dict(), c, torch.float32), strict=False)
    assert not unexpected and all("lm_head" in k for k in missing), (missing, unexpected)
    hf.tie_weights(); x = torch.randint(0, 128, (2, 16))
    with torch.no_grad(): assert torch.allclose(m(x), hf(x).logits, atol=1e-4, rtol=1e-3)
