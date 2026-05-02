"""
Baseline: target-model-only generation (no speculative decoding).
Used by the judge to measure correctness and speedup.
"""

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

TARGET_MODEL_PATH = os.getenv("TARGET_MODEL_PATH", "/models/gpt2-large")

_model     = None
_tokenizer = None


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    device = _get_device()
    src    = TARGET_MODEL_PATH if os.path.isdir(TARGET_MODEL_PATH) else "gpt2-large"
    _tokenizer = GPT2Tokenizer.from_pretrained(src)
    _model     = GPT2LMHeadModel.from_pretrained(src).to(device).eval()


def baseline_decode(
    prompt: str,
    max_new_tokens: int = 50,
    seed: int = 42,
) -> str:
    """Standard autoregressive greedy generation using only the target model."""
    _load()
    device    = next(_model.parameters()).device
    input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = _model(generated).logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated  = torch.cat([generated, next_token], dim=-1)

    return _tokenizer.decode(generated[0], skip_special_tokens=True)
