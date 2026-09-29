# Plan: contrastive encoder, Qwen 0.5B migration, other models under 5B

Status: plan (2026-09-28). Hardware: GTX 1650 Ti, 4 GB, Windows, commit limit ~35 GB.
Guiding result so far: on GPT-2 the NLA reveals planted concepts the text and the output don't show (iteration 8). The limit is information content, which grows with model quality and training steps.

---

## A. Contrastive activation encoder (wiki: `contrastive_activation_encoder`)

**Goal:** a second, independent reader that scores "which description fits this state?"
- It serves as corroboration and as the backend for fixed stakeholder questions (Noul / Choice / Score, as in CLM-8B).
- It needs no generation step.

**Design** (CLIP / CLM-8B recipe, sized for 4 GB):
- **State side:** the activation itself (d = 768 for GPT-2, 896 for Qwen 0.5B) → a 2-layer MLP head → 384-d, L2-normalised. No encoder pass is needed.
- **Text side:** frozen MiniLM (`all-MiniLM-L6-v2`, cached), optionally with a small linear adapter.
- **Loss:** bidirectional InfoNCE, with a learned temperature (CLIP).
- **Batches:** 256–1024 on the GPU, which is cheap because the state side is just vectors.

**Training data**, in order of increasing "internal-ness", tagged by source so evaluation can split on it:
1. `(activation, RL explanation)` pairs sampled from the rl2 AV. These are about the *state* and are the default.
2. `(activation, warm-start summary)` pairs. These are about the *text*; use them as a sanity baseline only.
3. `(steered activation, concept label)` pairs from `hidden_state_nla`-style steering with a *held-out* concept set. These teach "internal" concepts.
4. For models with SAEs (Gemma-2-2B via Gemma Scope, Qwen3 / Qwen3.5 via Qwen-Scope): `(activation, labels of its top SAE features)`. This is the self-interpretation-adapter recipe (arXiv 2602.10352).

**Files:**
- `nla/encoder.py`: the `ActivationEncoder` head, `score(acts, texts)`, save/load.
- `training/train_encoder.py`: InfoNCE training that logs retrieval@1 against shuffled descriptions.
- `training/eval_encoder.py`: runs the checks below.
- `NLAReader.ask(prompt, questions)` returns typed answers.
- A GUI "Ask" card on `/report`.

**Evaluation** (all against controls):
- **Held-out retrieval@1 / @5** among 100 candidates, compared with the NLA-text→MiniLM route. It should beat shuffled descriptions by a wide margin.
- **Planted-concept benchmark** (iteration 8) with concepts *not* in training, compared with the NLA and the lens at strengths 0.125–1.0. It must stay near chance on the random-direction placebo.
- **Hint benchmark** (iteration 9) with the contrast of encoder scores (hinted − baseline).
- **Calibration:** reliability diagrams and ECE for the Noul probabilities, with temperature scaling on the validation split. Stakeholders see buckets, not raw probabilities.
- **Guardrail:** every Choice question includes "none of these" plus two decoy options drawn from other items.

**Effort and cost:**
- About 1 day of code.
- Training takes minutes (GPT-2) to about 1 h (with steered data). The data generation for source 1, sampling AV explanations for about 20k activations, takes about 2 h of GPU time.

**Exit criteria:** planted-concept top-1 ≥ the NLA at faint strength, placebo ≤ 0.15, and calibrated ECE ≤ 0.05.

---

## B. Migration to Qwen2.5-0.5B-Instruct (config exists: `configs/qwen05b.yaml`)

**Why:** it's a chat model, so it can be asked questions, has more knowledge than GPT-2, and runs the same pipeline with LoRA. It's the step towards "what was it thinking when it *answered*".

**Blocking changes** (code):
1. **One base, many adapters.** Today `frozen_reference` deep-copies the AV (+1 GB), and the target, AV and AR each load their own weights.
   - With PEFT, use `av.lm.disable_adapter()` for the KL reference and for plain target forwards, and give the AV and AR named adapters on a shared base.
   - This saves about 2 GB at 0.5B and is *required* for anything ≥1.5B.
2. **Chat-template-aware prompts.**
   - The AV and AR prompts should go through `tokenizer.apply_chat_template` when the tokenizer has one (the `templates:` block stays as a fallback).
   - Datagen should capture activations on *assistant* turns too, so the NLA sees answering states, not just web text.
3. **Adapter coverage tests** on tiny models (`sshleifer/tiny-gpt2`, `hf-internal-testing/tiny-random-Qwen2ForCausalLM`, `...LlamaForCausalLM`, `...Gemma2ForCausalLM`), each running a 10-step pipeline on CPU.
4. **Lens:** there's no pretrained tuned lens for Qwen, and `Lens` already falls back to the logit lens. Optionally train one with the `tuned-lens` package (about 1 h at 0.5B).
5. **bf16/fp16 safety:** the 1650 Ti has no bf16, so keep fp16 base + fp32 LoRA and loss scaling (already present). Verify there are no overflow NaNs in the AR head at d = 896.

**Run plan:**
1. Datagen, 20k contexts (the explainer is the same Qwen model; ~3 h).
2. AR SFT (~2 h).
3. AV SFT (~3 h).
4. RL, 1,500 steps (estimated 12–18 h; GPT-2 took ~4.5 h).
5. eval → calibrate → anticipation → hidden_state → contrast.
6. **The dashboard needs no change:** it's config-driven, via `--config configs/qwen05b.yaml`.

**Success:**
- all touchstone checks pass;
- AV FVE above the SFT-AR summary baseline;
- a planted-concept hit rate ≥ GPT-2's at equal relative strength.

**Risks:**
- RL wall-clock time, so run overnight and resume with `rl.init_from`.
- A same-model explainer gives weak warm-start labels. A Claude explainer would help but needs an API key and the user's go-ahead.

---

## C. Other models under 5B: shortlist

| Model | Size | Why it matters here | Fits 4 GB? | Extras |
|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | 0.5B | first migration; cached | fp16 ✓ | logit lens |
| **Qwen3-0.6B / 1.7B** | 0.6 / 1.7B | **thinking mode**: visible reasoning chains, enabling the say-vs-think check (idea 9, the original goal) | 0.6B fp16 ✓; 1.7B 4-bit | Qwen-Scope SAEs (Qwen3/3.5 series) |
| **DeepSeek-R1-Distill-Qwen-1.5B** | 1.5B | long explicit chains of thought; the classic unfaithful-CoT test bed | 4-bit ✓ | — |
| **Gemma-2-2B-it** | 2.6B | **Gemma Scope SAEs** at every layer plus Neuronpedia labels: ground truth for the encoder and a third corroborating view | 4-bit ✓ (tight) | SAEs, Neuronpedia |
| **SmolLM2-360M / SmolLM3-3B** | 0.36 / 3B | fully open training data (checks "is it from the text?"); fast 360M iterations | ✓ / 4-bit tight | — |
| **Llama-3.2-1B / 3B-Instruct** | 1 / 3B | covers the Llama family; widely used | 1B fp16 ✓; 3B 4-bit tight | gated licence |
| **Pythia-410m / 1.4b** | 0.4 / 1.4B | **pretrained tuned lenses** (confirmed in the AlignmentResearch space); training checkpoints | ✓ | tuned lens |
| Phi-4-mini, Gemma-3-4B, Qwen3-4B / 3.5-4B | 3.8–4B | the strongest reasoning under 5B | 4-bit, very tight: only with one shared base and short contexts | — |

**Recommended order after Qwen 0.5B:**
1. **Qwen3-0.6B.** Reasoning chains at a size that fits fp16, which unlocks say-vs-think.
2. **Gemma-2-2B (4-bit).** SAE ground truth for the encoder and a triangulated report.
3. **DeepSeek-R1-Distill-Qwen-1.5B or Qwen3-1.7B.** Faithfulness of longer reasoning.

The 3–4B models come only after the shared-base refactor proves stable at 1.5B.

**Common prerequisites for ≥1.5B:**
- the shared-base refactor (B1);
- 4-bit base via bitsandbytes (QLoRA; now supported on Windows);
- gradient checkpointing on the AV;
- `max_context_tokens` 64–96;
- RL `batch_size` 1 × `group_size` 4.

Expect RL of 1–3 days per model on the 1650 Ti, so a cloud GPU (one 24 GB card) is the realistic path for 3–4B. That's a decision for the user.

---

## Order of work
0. ✅ **"Unusual state" flag** (iteration 10: `nla/unusual.py` + "unverified but consistent" themes; concept visible 85%, was 0%). Was: The reliability filter marks reads of unusual states as unreliable even when their words are right. Detect out-of-distribution activations (norm / Mahalanobis vs train) and tell the user "unusual state; explanation unverifiable", not "no signal".
1. Iteration 9 results → wiki (done).
2. **B1–B3** (shared base, chat templates, tiny-model tests). This helps every later model.
3. **A** (encoder) on GPT-2 rl2 data while the Qwen 0.5B datagen runs on the GPU.
4. **B** run (overnight RL).
5. The encoder on Qwen 0.5B.
6. **C** in the recommended order.

## Sources
- Released NLAs: only 7B+ (kitft: Qwen2.5-7B, Gemma-3-12B/27B, Llama-3.3-70B). The Qwen-7B RL used 16× H100, so there's no small-model reference.
- Tuned-lens space: GPT-2 family, Pythia 70m–12b, OPT-125m/1.3b; no Qwen, Llama-3.2 or Gemma.
- SAEs: Gemma Scope (Gemma-2-2B/9B), Gemma Scope 2 (Gemma 3), Qwen-Scope (arXiv 2605.11887; Qwen3/3.5), Neuronpedia.
- CLM-8B (MarkTechPost 2026-09-23); self-interpretation adapters (arXiv 2602.10352); Contrastive Projection (arXiv 2609.09902).
