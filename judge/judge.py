"""
Judge for the Speculative Decoding RL Environment
==================================================
Evaluates a candidate implementation at /solution/speculative_decoding.py

Scoring
-------
  FAIL  -> score 0.0    (hard failure: missing file, wrong signature, crash,
                          correctness below threshold, speedup out of range)
  PASS  -> continuous score in [0, 1] based on correctness + speedup

Steps
-----
  1. Check file exists + correct function signature
  2. Correctness: token match rate >= 95% vs greedy target-only baseline
  3. Speed: wall-clock speedup >= 1.5x vs baseline (use --local for 1.05x on CPU)
  4. Sanity: speedup <= 50x (flags output caching / hardcoded lookup tables)
  5. Compute continuous score

Note on eval prompts
--------------------
Prompts are embedded directly in this file rather than stored in a separate
data file. In deployment the judge runs in a restricted VM directory that the
LLM agent cannot read, preventing the agent from precomputing and caching
outputs for known prompts.
"""

import sys
import os
import time
import importlib.util
import inspect
import argparse

SOLUTION_PATH   = os.getenv("SOLUTION_PATH", "solution/speculative_decoding.py")
MAX_NEW_TOKENS  = 50
K               = 4
SEED            = 42
MATCH_THRESHOLD = 0.95
SPEEDUP_MIN_DEFAULT = 1.5
SPEEDUP_MAX_SCORE   = 3.0
SPEEDUP_SANITY      = 50.0   # above this strongly suggests output caching

# ---------------------------------------------------------------------------
# Eval prompts — embedded here so the LLM agent cannot read them from a file.
# ---------------------------------------------------------------------------
_EVAL_PROMPTS = [
    # Geography
    "The capital of France is", "The longest river in Africa is",
    "The highest mountain in the world is", "The capital of Japan is",
    "The Amazon rainforest is located in", "The Sahara Desert spans across",
    "The capital of Australia is", "The Pacific Ocean is the",
    "The capital of Brazil is", "The Nile River flows through",
    # Science
    "The speed of light is approximately", "Photosynthesis is the process by which",
    "The human body contains approximately", "The periodic table was invented by",
    "DNA stands for", "The theory of evolution was proposed by",
    "The boiling point of water at sea level is", "Gravity was described by",
    "The smallest unit of matter is", "The Big Bang theory states that",
    # History
    "The French Revolution began in", "World War II ended in",
    "The first moon landing occurred in", "The Roman Empire fell in",
    "The printing press was invented by", "The Declaration of Independence was signed in",
    "The Renaissance period began in", "The Cold War lasted from",
    "The Berlin Wall fell in", "The first computer was built in",
    # Technology
    "The first programming language was", "The internet was invented in",
    "Artificial intelligence refers to", "Machine learning is a subset of",
    "The Python programming language was created by", "The first iPhone was released in",
    "Cloud computing refers to", "The Linux kernel was created by",
    "Blockchain technology was first described in", "The transistor was invented in",
    # Math
    "The square root of 144 is", "Pi is approximately equal to",
    "The Fibonacci sequence begins with", "A prime number is defined as",
    "The Pythagorean theorem states that", "Calculus was invented by",
    "The value of Euler's number e is approximately", "A quadratic equation has the form",
    "The sum of angles in a triangle is", "Binary number system uses only",
    # Story starters
    "On a cold winter morning,", "The old lighthouse stood at the edge of",
    "She had never seen anything like it before,", "The last train left the station at",
    "Deep in the forest, there was a", "The scientist stared at the results and",
    "After ten years away,", "The message arrived at midnight,",
    "No one expected the discovery of", "The robot looked up and said,",
    # Nature
    "The migration of birds is triggered by", "Coral reefs are important because",
    "The water cycle consists of", "Volcanoes form when",
    "Earthquakes are caused by", "The Amazon produces approximately",
    "Polar ice caps are melting because", "The ozone layer protects Earth from",
    "Hurricanes form over", "Bioluminescence is the ability of",
    # Economics
    "Inflation refers to the", "The stock market is a place where",
    "Gross domestic product measures", "Supply and demand determines",
    "A recession is defined as", "Central banks control",
    "Cryptocurrency is a form of", "The gold standard refers to",
    "Free trade agreements allow", "Microeconomics focuses on",
    # Medicine
    "The human immune system protects", "Antibiotics are used to treat",
    "The heart pumps blood through", "Vaccines work by",
    "The nervous system is responsible for", "Cancer occurs when",
    "Mental health refers to", "The digestive system breaks down",
    "Genetics is the study of", "The placebo effect occurs when",
    # Space
    "The Milky Way galaxy contains", "Black holes are formed when",
    "The International Space Station orbits", "Mars is known as",
    "The speed required to escape Earth's gravity is", "Neutron stars are created when",
    "The James Webb Space Telescope can observe", "Solar flares are caused by",
    "The nearest star to Earth is", "Dark matter makes up approximately",
]


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


def run_judge(speedup_min: float = SPEEDUP_MIN_DEFAULT):
    print("=" * 60)
    print("Speculative Decoding Environment — Judge")
    print(f"Speedup threshold: {speedup_min}x  |  Sanity cap: {SPEEDUP_SANITY}x")
    print("=" * 60)

    print("\n[1/4] Checking file and function signature...")
    candidate_fn = load_candidate()
    print("      OK")

    print("\n[2/4] Loading baseline (target-model-only)...")
    baseline_fn = load_baseline()
    print("      OK")

    prompts = _EVAL_PROMPTS
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
    print(f"      Speedup          : {speedup:.2f}x   (range: >= {speedup_min}x, <= {SPEEDUP_SANITY}x)")
    print(f"      Baseline time    : {t_base:.1f}s")
    print(f"      Candidate time   : {t_cand:.1f}s")

    if match_rate < MATCH_THRESHOLD:
        fail(f"Correctness check failed: {match_rate:.4f} < {MATCH_THRESHOLD}. "
             f"Output does not match greedy target-only generation.")

    if speedup < speedup_min:
        fail(f"Speed check failed: {speedup:.2f}x < {speedup_min}x. "
             f"Ensure the target model is called ONCE per K-token round.")

    if speedup > SPEEDUP_SANITY:
        fail(f"Sanity check failed: {speedup:.1f}x > {SPEEDUP_SANITY}x. "
             f"Suspiciously fast — output caching or hardcoded results detected.")

    correctness_score = min(1.0, max(0.0, (match_rate - MATCH_THRESHOLD) / (1.0 - MATCH_THRESHOLD)))
    speedup_score     = min(1.0, max(0.0, (speedup - speedup_min) / (SPEEDUP_MAX_SCORE - speedup_min)))
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
    run_judge(speedup_min=1.05 if args.local else SPEEDUP_MIN_DEFAULT)
