# TinyNLA

A small-model **Natural Language Autoencoder** (NLA) that can read a language model's "thoughts". It follows
[Fraser-Taliente et al. 2026](https://transformer-circuits.pub/2026/nla/) and
[kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders), scaled down to a single 4 GB GPU.

```
activation h (one token, layer K) ──► AV ──► "explanation" ──► AR ──► ĥ
                                     verbalizer           reconstructor
score: FVE = 1 − E‖h − ĥ‖² / E‖h − h̄‖²      (0 = no better than the average activation)
```

- **AV (activation verbalizer)**: a copy of the target model. The activation is injected as the embedding of a special
  `<|inject|>` token in `Explain: <concept><|inject|></concept>`, and the AV writes `<explanation>…</explanation>`.
- **AR (activation reconstructor)**: the target's own first K+1 blocks plus a `Linear(d,d)` head, read out at the last token.
- **Training**:
  1. Warm start: SFT on summaries of the text up to the token.
  2. **GRPO**: the AV is rewarded for explanations that let the AR rebuild the activation.

  RL is what makes explanations describe the internal state rather than just restating the input.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Run the pipeline

```powershell
.\run_nla.ps1                                   # GPT-2 small, layer 8
.\run_nla.ps1 -From av_sft                      # resume from a stage
.\run_nla.ps1 -Config configs\qwen05b.yaml      # Qwen2.5-0.5B-Instruct with LoRA
```

| Stage | Command | Output |
|---|---|---|
| Data | `python -m training.datagen` | `data/<run>/buffer.pt`, `split.json`, `nla_meta.yaml` |
| AR SFT | `python -m training.train_ar_sft` | `checkpoints/<run>/ar_sft/` + `ar_sft_eval.json` |
| AV SFT | `python -m training.train_av_sft` | `checkpoints/<run>/av_sft/` (sets `injection_scale`) |
| RL | `python -m training.train_rl` | `checkpoints/<run>/rl/{av,ar}` + `metrics.jsonl` |
| Eval | `python -m training.eval_nla` | `nla_eval.json` next to the evaluated AV |

Every entry point takes `--config` and repeatable `--set section.key=value` overrides
(e.g. `--set rl.steps=500 --set device=cpu`).

## Read thoughts

```powershell
python -m nla.read "The capital of France is" --samples 3
python -m nla.gui                                # Thought Reader at http://127.0.0.1:8000
```

The GUI tokenizes your text. Click a token (or use ←/→ and Enter) to stream explanations for its activation.

- Each explanation has an AR self-check FVE.
- Words that recur across samples are highlighted, since recurring claims are more trustworthy.
- A chart compares the result against the controls: shuffled explanations and the average activation.
- Clicking an explanation shows the true vs reconstructed activation on the dimensions where they differ most.

## Is it working? (touchstone checks)

`eval_nla` reports these on held-out activations:

| Check | Pass condition |
|---|---|
| 1 beats the mean | `av_fve` > 0.05 |
| 1b beats the input summary | `av_fve` > `summary_fve` (after RL; the evidence of reading internal state) |
| 2 shuffle control | `av_fve_shuffled` ≤ ~0 and well below `av_fve` |
| 5 diverse explanations | `unique_frac` > 0.9 |

Cosine similarity alone is not evidence: GPT-2 activations are anisotropic, so a constant vector already scores about 0.8.

## Layout

```
nla/            model_adapter, datagen, explainers, ar, av, rl, metrics, evaluate, runs, read, gui/
training/       datagen, train_ar_sft, train_av_sft, train_rl, eval_nla
configs/        gpt2_small.yaml, qwen05b.yaml   (base.yaml = legacy)
tests/          unit tests (fast) + tests/test_e2e_smoke.py (-m slow)
nla/legacy, training/legacy, docs/legacy_v3.md  the v3 trajectory pipeline
```

## Tests

```powershell
.venv\Scripts\python -m pytest                   # fast unit tests
.venv\Scripts\python -m pytest -m slow -s        # end-to-end smoke run on CPU
```

## Hardware notes (GTX 1650 Ti, 4 GB)

GPT-2 small uses full fine-tuning in fp32 with AMP. RL fits the policy, a frozen fp16 reference and the AR in about 3.5 GB.

Qwen 0.5B needs LoRA with an fp16 base, because Turing GPUs have no bf16.

Datagen is dominated by the local explainer (about 0.65 s per sample), so 20k samples take about 3.5 hours. The cache is append-only JSONL, so interrupted runs resume where they stopped.
