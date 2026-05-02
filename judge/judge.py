"""
Judge for the Speculative Decoding RL Environment
==================================================
Evaluates a candidate implementation at /solution/speculative_decoding.py

Scoring
-------
  FAIL  -> score 0.0    (hard failure: missing file, wrong signature, crash)
  PASS  -> continuous score in [0, 1] based on correctness + speedup

Steps
-----
  1. Check file exists
  2. Check import + function signature
  3. Correctness: token match rate >= 95% on hidden eval prompts
  4. Speed: wall-clock speedup >= 1.5x vs baseline (use --local for 1.05x on CPU)
  5. Compute continuous score
"""

import sys
import os
import time
import importlib.util
import inspect
import argparse

SOLUTION_PATH   = os.getenv("SOLUTION_PATH", "solution/speculative_decoding.py")
EVAL_PROMPTS    = os.getenv("EVAL_PROMPTS",  "data/eval_prompts.txt")
MAX_NEW_TOKENS  = 50
K               = 4
SEED            = 42
MATCH_THRESHOLD = 0.95
SPEEDUP_MAX     = 3.0


def fail(reason: str):
    print(f"\nFAIL: {reason}", file=sys.stderr)
    print("score=0.0")
    sys.exit(1)


def load_candidate():
    if not os.path.isfile(SOLUTION_PATH):
        fail(f"File not found: {SOLUTION_PATH}")

    spec   = importlib.util.spec_from_file_location("candidate", SOLUTION_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        fail(f"Import error: {e}")

    if not hasattr(module, "speculative_decode"):
        fail("Function 'speculative_decode' not found in solution file.")

    fn  = module.speculative_decode
    sig = inspect.signature(fn)
    expected = {"prompt", "max_new_tokens", "K", "seed"}
    missing  = expected - set(sig.parameters.keys())
    if missing:
        fail(f"Missing parameters in speculative_decode: {missing}")

    return fn


def load_baseline():
    path   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../solution/baseline.py")
    spec   = importlib.util.spec_from_file_location("baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.baseline_decode


def load_prompts():
    if not os.path.isfile(EVAL_PROMPTS):
        fail(f"Eval prompts not found: {EVAL_PROMPTS}. Run: python data/generate_eval_prompts.py")
    with open(EVAL_PROMPTS) as f:
        prompts = [l.strip() for l in f if l.strip()]
    if len(prompts) < 10:
        fail("Eval prompts file has fewer than 10 prompts.")
    return prompts


def run_judge(speedup_min: float = 1.5):
    print("=" * 60)
    print("Speculative Decoding Environment — Judge")
    print(f"Speedup threshold: {speedup_min}x")
    print("=" * 60)

    print("\n[1/4] Checking file and function signature...")
    candidate_fn = load_candidate()
    print("      OK")

    print("\n[2/4] Loading baseline (target-model-only)...")
    baseline_fn = load_baseline()
    print("      OK")

    prompts = load_prompts()
    print(f"\n[3/4] Evaluating on {len(prompts)} prompts...")

    from transformers import GPT2Tokenizer
    tok_src   = "/models/gpt2-small" if os.path.isdir("/models/gpt2-small") else "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(tok_src)

    n_total   = 0
    n_matched = 0
    t_cand    = 0.0
    t_base    = 0.0

    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        try:
            ref = baseline_fn(prompt=prompt, max_new_tokens=MAX_NEW_TOKENS, seed=SEED)
        except Exception as e:
            fail(f"Baseline crashed on prompt {i}: {e}")
        t_base += time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            cand = candidate_fn(prompt=prompt, max_new_tokens=MAX_NEW_TOKENS, K=K, seed=SEED)
        except Exception as e:
            fail(f"Candidate crashed on prompt {i}: {e}")
        t_cand += time.perf_counter() - t0

        prompt_len  = len(tokenizer.encode(prompt))
        ref_tokens  = tokenizer.encode(ref) [prompt_len:]
        cand_tokens = tokenizer.encode(cand)[prompt_len:]
        min_len     = min(len(ref_tokens), len(cand_tokens))

        if min_len > 0:
            n_matched += sum(r == c for r, c in zip(ref_tokens[:min_len], cand_tokens[:min_len]))
            n_total   += min_len

        if (i + 1) % 10 == 0:
            rate = n_matched / max(n_total, 1)
            print(f"      Prompt {i+1:3d}/{len(prompts)} — running match rate: {rate:.3f}")

    match_rate = n_matched / max(n_total, 1)
    speedup    = t_base / max(t_cand, 1e-6)

    print(f"\n[4/4] Final results:")
    print(f"      Token match rate : {match_rate:.4f}  (threshold >= {MATCH_THRESHOLD})")
    print(f"      Speedup          : {speedup:.2f}x   (threshold >= {speedup_min}x)")
    print(f"      Baseline time    : {t_base:.1f}s")
    print(f"      Candidate time   : {t_cand:.1f}s")

    if match_rate < MATCH_THRESHOLD:
        fail(f"Correctness check failed: {match_rate:.4f} < {MATCH_THRESHOLD}. "
             f"Check your rejection sampling math.")

    if speedup < speedup_min:
        fail(f"Speed check failed: {speedup:.2f}x < {speedup_min}x. "
             f"Ensure target model is called ONCE per K-token round.")

    correctness_score = min(1.0, max(0.0, (match_rate - MATCH_THRESHOLD) / (1.0 - MATCH_THRESHOLD)))
    speedup_score     = min(1.0, max(0.0, (speedup - speedup_min) / (SPEEDUP_MAX - speedup_min)))
    final_score       = 0.6 * correctness_score + 0.4 * speedup_score

    print(f"\n{'='*60}")
    print(f"  PASS")
    print(f"  Correctness component : {correctness_score:.4f}  (weight 0.6)")
    print(f"  Speedup component     : {speedup_score:.4f}  (weight 0.4)")
    print(f"  Final score           : {final_score:.4f}")
    print(f"{'='*60}")
    print(f"score={final_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true",
                        help="Lower speedup threshold to 1.05x for CPU/local testing")
    args = parser.parse_args()
    run_judge(speedup_min=1.05 if args.local else 1.5)
