"""
Speedup benchmark tests.
Run with: pytest tests/test_speedup.py -v -s
(The -s flag shows print output including speedup ratios)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pytest
from solution.speculative_decoding import speculative_decode
from solution.baseline import baseline_decode

PROMPTS = [
    "The theory of relativity explains",
    "In the beginning of the universe",
    "Artificial intelligence will",
]


def measure_time(fn, **kwargs):
    start  = time.perf_counter()
    result = fn(**kwargs)
    return result, time.perf_counter() - start


def test_speculative_is_faster_than_baseline():
    """
    Speculative decoding should be faster than baseline.
    On GPU: expect 2-3x. On CPU: expect 1.1-1.4x.
    This test passes at any speedup > 1.0.
    """
    t_spec = 0.0
    t_base = 0.0

    for prompt in PROMPTS:
        _, tb = measure_time(baseline_decode, prompt=prompt, max_new_tokens=30, seed=42)
        _, ts = measure_time(speculative_decode, prompt=prompt, max_new_tokens=30, K=4, seed=42)
        t_base += tb
        t_spec += ts

    speedup = t_base / max(t_spec, 1e-6)
    print(f"\n  Baseline total : {t_base:.2f}s")
    print(f"  Speculative    : {t_spec:.2f}s")
    print(f"  Speedup        : {speedup:.2f}x")

    assert speedup > 1.0, (
        f"Speculative decoding ({t_spec:.2f}s) was not faster than "
        f"baseline ({t_base:.2f}s). Speedup: {speedup:.2f}x"
    )


def test_speedup_scales_with_k():
    """
    Higher K should generally give higher speedup (up to a point).
    K=4 should be faster than K=1 on average.
    """
    prompt = "The future of artificial intelligence"

    _, t1 = measure_time(speculative_decode, prompt=prompt, max_new_tokens=30, K=1, seed=42)
    _, t4 = measure_time(speculative_decode, prompt=prompt, max_new_tokens=30, K=4, seed=42)

    print(f"\n  K=1 time: {t1:.2f}s")
    print(f"  K=4 time: {t4:.2f}s")
    print(f"  K=4 vs K=1 speedup: {t1/max(t4,1e-6):.2f}x")

    # K=4 should be at least as fast as K=1 (allowing 10% slack)
    assert t4 <= t1 * 1.1, (
        f"K=4 ({t4:.2f}s) was significantly slower than K=1 ({t1:.2f}s). "
        f"The target model may be called too many times."
    )
