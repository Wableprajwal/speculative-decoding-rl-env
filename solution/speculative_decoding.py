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
  1. Use the small draft model to auto-regressively propose K tokens cheaply.
  2. Feed the original context + all K draft tokens into the target model in
     ONE forward pass (the target processes all positions in parallel).
  3. Accept or reject each draft token via rejection sampling, which
     guarantees the final distribution is identical to target-only generation.
  4. If a token is rejected, resample from a corrected distribution and
     discard all subsequent draft tokens.

This typically achieves 2-3x wall-clock speedup because:
  - The target model's forward pass cost scales sub-linearly with extra tokens
    (attention is parallelised over the sequence).
  - The draft model is ~6x smaller, so its K sequential calls are cheap.
"""

import os
import torch
import numpy as np
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
    Generate text using speculative decoding.

    Parameters
    ----------
    prompt         : Input text string.
    max_new_tokens : How many new tokens to generate beyond the prompt.
    K              : Speculation length — tokens proposed per draft round.
    seed           : Random seed for reproducibility.

    Returns
    -------
    Full string: prompt + generated continuation.
    """
    _load_models()
    torch.manual_seed(seed)
    np.random.seed(seed)

    device    = next(_target_model.parameters()).device
    input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated   = input_ids.clone()
    n_generated = 0

    while n_generated < max_new_tokens:
        remaining = max_new_tokens - n_generated
        k = min(K, remaining)

        # Step 1: Draft model proposes k tokens autoregressively (cheap)
        # Store full distributions (not just chosen-token scalar) for correct
        # rejection-sampling correction: normalize(max(0, p_target - p_draft))
        draft_ids   = []
        draft_dists = []  # full vocab distributions, shape (vocab_size,) each
        ctx = generated.clone()

        for _ in range(k):
            logits     = _get_logits(_draft_model, ctx)
            probs      = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            draft_ids.append(next_token.item())
            draft_dists.append(probs[0])  # full distribution over vocab
            ctx = torch.cat([ctx, next_token], dim=-1)

        # Step 2: Target model verifies ALL k tokens in ONE forward pass
        draft_tensor = torch.tensor([draft_ids], device=device)
        full_ctx     = torch.cat([generated, draft_tensor], dim=-1)
        all_logits   = _get_logits(_target_model, full_ctx)
        base_pos     = generated.shape[1] - 1

        # Step 3: Rejection sampling
        # Accept/reject each draft token; on rejection sample corrected distribution
        accepted = 0

        for i in range(k):
            tgt_probs = torch.softmax(all_logits[:, base_pos + i, :], dim=-1)
            token_id  = draft_ids[i]
            p_target  = tgt_probs[0, token_id].item()
            p_draft   = draft_dists[i][token_id].item()

            u = torch.rand(1).item()
            if u <= min(1.0, p_target / (p_draft + 1e-10)):
                # Accept
                generated   = torch.cat([generated, torch.tensor([[token_id]], device=device)], dim=-1)
                accepted    += 1
                n_generated += 1
                if n_generated >= max_new_tokens:
                    break
            else:
                # Reject: resample from corrected distribution
                # Subtract full draft dist from target dist, clamp negatives to 0
                corrected = torch.clamp(tgt_probs[0] - draft_dists[i], min=0.0)
                s = corrected.sum()
                corrected = corrected / s if s > 0 else tgt_probs[0]
                token     = torch.multinomial(corrected, num_samples=1)
                generated   = torch.cat([generated, token.unsqueeze(0)], dim=-1)
                n_generated += 1
                break

        # Bonus token when all k draft tokens accepted (standard protocol)
        if accepted == k and n_generated < max_new_tokens:
            bonus_probs = torch.softmax(all_logits[:, base_pos + k, :], dim=-1)
            bonus_token = torch.multinomial(bonus_probs, num_samples=1)
            generated   = torch.cat([generated, bonus_token], dim=-1)
            n_generated += 1

    return _tokenizer.decode(generated[0], skip_special_tokens=True)
