# Speculative Decoding — RL Environment for LLM Training

A complete RL environment designed to train LLMs to implement
**speculative decoding** — a key inference acceleration technique used in
production at Google, Meta, and Anthropic.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wableprajwal/speculative-decoding-rl-env/blob/main/Run_on_Colab.ipynb)

## What is Speculative Decoding?

Standard LLM generation calls the large target model once per token —
sequential and slow. Speculative decoding uses a small, fast **draft model**
to propose K tokens at once, then uses the large **target model** to verify
all K tokens in a **single parallel forward pass** via rejection sampling.

**Result: 2–3x wall-clock speedup with zero output quality loss.**
The output distribution is mathematically identical to target-only generation.

```
Input prompt
     │
     ▼
Draft model (GPT-2 small) ── proposes K tokens cheaply
     │
     ▼
Target model (GPT-2 large) ─ verifies ALL K tokens in ONE forward pass
     │
     ├── Accept tokens matching target distribution
     └── On first rejection: resample from corrected distribution, discard rest
```

## Quick Start

**Easiest — run on Google Colab (free T4 GPU, no setup):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wableprajwal/speculative-decoding-rl-env/blob/main/Run_on_Colab.ipynb)

**Local (CPU — models download ~1.5GB on first run):**

```bash
pip install -r requirements.txt
python data/generate_eval_prompts.py
python scripts/smoke_test.py
pytest tests/ -v
python judge/judge.py --local   # --local lowers speedup threshold for CPU
```

## Repo Structure

```
├── solution/
│   ├── speculative_decoding.py   # Reference implementation
│   └── baseline.py               # Target-model-only baseline
├── judge/
│   └── judge.py                  # Automated judge (correctness + speedup)
├── data/
│   ├── sample_prompts.txt        # 5 prompts for development/testing
│   └── generate_eval_prompts.py  # Generates 100 hidden eval prompts
├── tests/
│   ├── test_correctness.py       # Algorithm correctness unit tests
│   └── test_speedup.py           # Speedup benchmark tests
├── scripts/
│   └── smoke_test.py             # Quick side-by-side comparison
└── Run_on_Colab.ipynb            # Full GPU demo notebook
```

## The Task (Prompt given to the LLM agent)

The LLM is asked to produce `/solution/speculative_decoding.py` containing:

```python
def speculative_decode(
    prompt: str,
    max_new_tokens: int = 50,
    K: int = 4,
    seed: int = 42,
) -> str:
    ...
```

**Requirements:**
1. Implement speculative decoding using the rejection sampling procedure from [Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192)
2. Output distribution must be statistically identical to target-model-only generation (≥ 95% token match on 100 hidden prompts)
3. Achieve ≥ 1.5× wall-clock speedup over target-model-only generation at K=4

## Judge Logic

| Step | Check | Failure condition |
|------|-------|-------------------|
| 1 | File exists | `/solution/speculative_decoding.py` missing |
| 2 | Import + signature | Import error or wrong parameters |
| 3 | Correctness | Token match rate < 95% on 100 hidden prompts |
| 4 | Speed | Speedup < 1.5× vs. baseline |

**Continuous score (if all checks pass):**
```python
correctness_score = (match_rate - 0.95) / 0.05   # 0.95→0.0, 1.00→1.0
speedup_score     = (speedup - 1.5) / 1.5         # 1.5x→0.0, 3.0x→1.0
final_score       = 0.6 * correctness_score + 0.4 * speedup_score
```

## Reward Hacking Analysis

| Potential hack | Why it fails |
|---|---|
| Hardcode outputs | Judge uses 100 hidden, unseen prompts |
| Skip draft model (run target only) | Passes correctness, **fails speedup gate** |
| Skip target model (run draft only) | Passes speed, **fails 95% correctness gate** |
| Call target K times instead of once | Correct output but no speedup → **fails** |

**Key design:** Correctness and speed are in fundamental tension — the only
path to a high score is implementing the algorithm correctly.

## Core Algorithm

```
For each round:
  1. Draft model proposes K tokens: x_1, x_2, ..., x_K
  2. Target model scores all K+1 positions in ONE forward pass
  3. For each draft token x_i:
       p_accept = min(1, p_target(x_i) / p_draft(x_i))
       if uniform() <= p_accept: accept
       else: resample from normalize(max(0, p_target - p_draft)), stop round
  4. If all K accepted: sample one bonus token from target
```

**Reference:** Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. [arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)
