"""
Unit tests for speculative decoding correctness.
Run with: pytest tests/test_correctness.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from solution.speculative_decoding import speculative_decode
from solution.baseline import baseline_decode


def test_returns_string():
    """Output should be a string."""
    result = speculative_decode("The sky is", max_new_tokens=5, K=2, seed=42)
    assert isinstance(result, str)


def test_output_starts_with_prompt():
    """Generated output must contain the original prompt."""
    prompt = "The capital of France is"
    result = speculative_decode(prompt, max_new_tokens=10, K=2, seed=42)
    assert result.startswith(prompt), f"Output did not start with prompt.\nGot: {result}"


def test_deterministic_with_same_seed():
    """Same seed should produce identical output."""
    prompt = "Once upon a time"
    out1 = speculative_decode(prompt, max_new_tokens=15, K=4, seed=42)
    out2 = speculative_decode(prompt, max_new_tokens=15, K=4, seed=42)
    assert out1 == out2, "Output is not deterministic with the same seed."


def test_different_seeds_differ():
    """Different seeds should (almost always) produce different output."""
    prompt = "The universe began"
    out1 = speculative_decode(prompt, max_new_tokens=20, K=4, seed=1)
    out2 = speculative_decode(prompt, max_new_tokens=20, K=4, seed=99)
    assert out1 != out2, "Different seeds produced identical output (very unlikely if correct)."


def test_generates_correct_token_count():
    """Output should contain approximately max_new_tokens new tokens."""
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt    = "Machine learning is"
    n         = 20
    result    = speculative_decode(prompt, max_new_tokens=n, K=4, seed=42)
    prompt_len  = len(tokenizer.encode(prompt))
    result_len  = len(tokenizer.encode(result))
    new_tokens  = result_len - prompt_len
    # Allow small tolerance: between n-2 and n+2
    assert abs(new_tokens - n) <= 2, f"Expected ~{n} new tokens, got {new_tokens}."


def test_k1_close_to_baseline():
    """
    With K=1, speculative decoding is near-identical to baseline
    (each draft token is independently verified one at a time).
    Token match rate should be very high.
    """
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    prompts = [
        "The capital of Germany is",
        "Scientists recently discovered",
    ]

    total   = 0
    matched = 0
    for prompt in prompts:
        spec = speculative_decode(prompt, max_new_tokens=15, K=1, seed=42)
        base = baseline_decode(prompt, max_new_tokens=15, seed=42)
        pl   = len(tokenizer.encode(prompt))
        st   = tokenizer.encode(spec)[pl:]
        bt   = tokenizer.encode(base)[pl:]
        n    = min(len(st), len(bt))
        matched += sum(s == b for s, b in zip(st[:n], bt[:n]))
        total   += n

    match_rate = matched / max(total, 1)
    assert match_rate >= 0.85, (
        f"K=1 match rate {match_rate:.3f} too low — "
        f"rejection sampling may be incorrect."
    )


def test_no_empty_output():
    """Output should never be empty or just the prompt."""
    prompt = "Deep in the ocean"
    result = speculative_decode(prompt, max_new_tokens=10, K=4, seed=42)
    assert len(result) > len(prompt), "Output appears to contain no generated tokens."
