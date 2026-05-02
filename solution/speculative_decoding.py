"""
Speculative Decoding — Reference Implementation
================================================
Draft model  : GPT-2 small  (117M params)
Target model : GPT-2 large  (774M params)

Algorithm follows: Leviathan et al. (2023)
"Fast Inference from Transformers via Speculative Decoding"
https://arxiv.org/abs/2211.17192

Key idea
--------
Instead of calling the large target model once per token, we:
  1. Use the small draft model to greedily propose K tokens cheaply.
  2. Feed the original context + all K draft tokens into the target model in
     ONE forward pass (the target processes all positions in parallel).
  3. Accept each draft token deterministically: if the target's greedy choice
     at that position matches the draft token, accept; otherwise take the
     target's choice and discard all subsequent draft tokens.
  4. When all K tokens are accepted, take a free bonus token from the target's
     final position — giving K+1 tokens for one target call.

Using greedy (argmax) decoding in both models guarantees that the speculative
output is bit-for-bit identical to target-only greedy generation, which makes
token match rate verifiable and deterministic.

This typically achieves 2-3x wall-clock speedup because:
  - The target model's forward pass cost scales sub-linearly with extra tokens
    (attention is parallelised over the sequence).
  - The draft model is ~6x smaller, so its K sequential calls are cheap.
"""

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DRAFT_MODEL_PATH  = os.getenv("DRAFT_MODEL_PATH",  "/models/gpt2-small")
TARGET_MODEL_PATH = os.getenv("TARGET_MODEL_PATH", "/models/gpt2-large")

_draft_model  = None
_target_model = None
_tokenizer    = None


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_models():
    """Load both models once and cache them."""
    global _draft_model, _target_model, _tokenizer
    if _draft_model is not None:
        return

    device     = _get_device()
    draft_src  = DRAFT_MODEL_PATH  if os.path.isdir(DRAFT_MODEL_PATH)  else "gpt2"
    target_src = TARGET_MODEL_PATH if os.path.isdir(TARGET_MODEL_PATH) else "gpt2-large"

    print(f"Loading models on device: {device}")
    _tokenizer    = GPT2Tokenizer.from_pretrained(draft_src)
    _draft_model  = GPT2LMHeadModel.from_pretrained(draft_src).to(device).eval()
    _target_model = GPT2LMHeadModel.from_pretrained(target_src).to(device).eval()
    print("Models loaded.")


def _get_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Single forward pass; returns logits (1, seq_len, vocab_size)."""
    with torch.no_grad():
        return model(input_ids).logits


def speculative_decode(
    prompt: str,
    max_new_tokens: int = 50,
    K: int = 4,
    seed: int = 42,
) -> str:
    """
    Generate text using speculative decoding (greedy).

    Parameters
    ----------
    prompt         : Input text string.
    max_new_tokens : How many new tokens to generate beyond the prompt.
    K              : Speculation length — tokens proposed per draft round.
    seed           : Unused (kept for API compatibility; decoding is deterministic).

    Returns
    -------
    Full string: prompt + generated continuation.
    """
    _load_models()

    device    = next(_target_model.parameters()).device
    input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated   = input_ids.clone()
    n_generated = 0

    while n_generated < max_new_tokens:
        remaining = max_new_tokens - n_generated
        k = min(K, remaining)

        # Step 1: Draft model greedily proposes k tokens (cheap, sequential)
        draft_ids = []
        ctx = generated.clone()
        for _ in range(k):
            logits     = _get_logits(_draft_model, ctx)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            draft_ids.append(next_token.item())
            ctx = torch.cat([ctx, next_token], dim=-1)

        # Step 2: Target model verifies ALL k tokens in ONE forward pass
        draft_tensor = torch.tensor([draft_ids], device=device)
        full_ctx     = torch.cat([generated, draft_tensor], dim=-1)
        all_logits   = _get_logits(_target_model, full_ctx)
        base_pos     = generated.shape[1] - 1

        # Step 3: Deterministic acceptance
        # Accept if target's greedy choice matches the draft; else take target's
        # choice and end the round (discard remaining draft tokens).
        accepted = 0
        for i in range(k):
            target_token = all_logits[:, base_pos + i, :].argmax(dim=-1).item()
            generated    = torch.cat([generated, torch.tensor([[target_token]], device=device)], dim=-1)
            n_generated += 1
            if target_token == draft_ids[i]:
                accepted += 1
                if n_generated >= max_new_tokens:
                    break
            else:
                break

        # Bonus token when all k draft tokens were accepted (K+1 tokens per call)
        if accepted == k and n_generated < max_new_tokens:
            bonus_token = all_logits[:, base_pos + k, :].argmax(dim=-1).item()
            generated   = torch.cat([generated, torch.tensor([[bonus_token]], device=device)], dim=-1)
            n_generated += 1

    return _tokenizer.decode(generated[0], skip_special_tokens=True)
