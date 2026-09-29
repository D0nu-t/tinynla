"""
nla/av.py

Activation Verbalizer (AV): activation vector -> explanation text.

As in the published NLA, the AV is a copy of the target model. Its prompt
contains a special marker token; the marker's input embedding is replaced
by the activation, rescaled to a fixed L2 norm (`injection_scale`), and the
model then writes "<explanation> ... </explanation>".

Prompt (from nla_meta.yaml templates):
    "Explain: <concept><|inject|></concept>\n<explanation>"
<|inject|> is added to the tokenizer as a single special token, so there is
exactly one match and no neighbour check is needed (kitft needs one because
it reuses a rare existing token).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla import model_adapter as ma

INJECT_TOKEN = "<|inject|>"


def choose_injection_scale(
    activations: torch.Tensor,
    token_embedding_norms: torch.Tensor,
    d_model: int,
) -> float:
    """
    Pick the L2 norm the injected activation is rescaled to.

    Args:
        activations:            [N, d] raw training activations at block K
        token_embedding_norms:  [vocab] norms of the AV's input embeddings
        d_model:                hidden size

    Returns:
        the fixed norm every injected vector is scaled to (a float > 0)
    """
    # Inject at the typical residual norm of the layer being read, like the
    # released checkpoints (Qwen-7B uses 150, inside its 100-170 residual
    # range, not sqrt(d) = 60). A token-embedding-sized vector (~4 for GPT-2)
    # is only a faint signal among its neighbours; the residual scale makes
    # the position salient, and fine-tuning teaches layer 0 to read it.
    # The median ignores rare very-high-norm activations. d_model and the
    # embedding norms are kept in the signature for alternative policies.
    del token_embedding_norms, d_model
    return float(activations.float().norm(dim=-1).median())


class ActivationVerbalizer(nn.Module):
    def __init__(
        self,
        target_name: str,
        prompt: str,
        response_open: str,
        response_close: str,
        injection_scale: float,
        max_explanation_tokens: int = 64,
        model_path: Optional[str] = None,
        lora: Optional[dict] = None,
        base_dtype: str = "float32",
    ):
        super().__init__()
        self.target_name = target_name
        self.prompt = prompt
        self.response_open = response_open
        self.response_close = response_close
        self.injection_scale = float(injection_scale)
        self.max_explanation_tokens = max_explanation_tokens
        self.lora = lora
        self.base_dtype = base_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path or target_name)
        if INJECT_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [INJECT_TOKEN]})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = getattr(torch, base_dtype)
        if lora:
            # frozen base (from the hub) + trainable LoRA adapter (from model_path)
            self.lm = AutoModelForCausalLM.from_pretrained(target_name, dtype=dtype)
            # Only grow the table. Qwen ships spare rows (151,936 vs 151,665 tokens);
            # resizing to len(tokenizer) would SHRINK it and re-initialise rows on
            # every load. The <|inject|> row itself is never read (overwritten).
            if self.lm.get_input_embeddings().num_embeddings < len(self.tokenizer):
                self.lm.resize_token_embeddings(len(self.tokenizer))
            from peft import LoraConfig, PeftModel, get_peft_model
            if model_path:
                self.lm = PeftModel.from_pretrained(self.lm, model_path, is_trainable=True)
            else:
                self.lm = get_peft_model(self.lm, LoraConfig(task_type="CAUSAL_LM", **lora))
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(model_path or target_name, dtype=dtype)
            if self.lm.get_input_embeddings().num_embeddings < len(self.tokenizer):
                self.lm.resize_token_embeddings(len(self.tokenizer))

        self.embed_scale = ma.embed_scale(self.lm)
        self.eos_id = self.tokenizer.eos_token_id
        self.inject_id = self.tokenizer.convert_tokens_to_ids(INJECT_TOKEN)

        # generation starts right after "<explanation>"
        self.prompt_ids: List[int] = self.tokenizer(prompt + response_open)["input_ids"]
        inject_id = self.tokenizer.convert_tokens_to_ids(INJECT_TOKEN)
        hits = [i for i, t in enumerate(self.prompt_ids) if t == inject_id]
        if len(hits) != 1:
            raise ValueError(f"prompt must contain exactly one {INJECT_TOKEN}, found {len(hits)}")
        self.inject_pos = hits[0]

    @property
    def device(self) -> torch.device:
        return self.lm.get_input_embeddings().weight.device

    # ------------------------------------------------------------------
    # Embedding with injection
    # ------------------------------------------------------------------

    def _embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(ids) * self.embed_scale

    def embed_prompts(self, vectors: torch.Tensor) -> torch.Tensor:
        """[B, d] activations -> [B, P, d] prompt embeddings with injection."""
        b = vectors.shape[0]
        ids = torch.tensor(self.prompt_ids, device=self.device).expand(b, -1)
        emb = self._embed_ids(ids).clone()
        v = F.normalize(vectors.to(self.device).float(), dim=-1) * self.injection_scale
        emb[:, self.inject_pos] = v.to(emb.dtype)
        return emb

    # ------------------------------------------------------------------
    # Supervised training
    # ------------------------------------------------------------------

    def response_ids(self, explanation: str) -> List[int]:
        body = self.tokenizer(
            explanation, truncation=True, max_length=self.max_explanation_tokens
        )["input_ids"]
        return body + self.tokenizer(self.response_close)["input_ids"] + [self.eos_id]

    def sequence_logprobs(
        self,
        vectors: torch.Tensor,
        responses: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-token log-probs of `responses` given injected prompts.

        Returns:
            logprobs [B, R] and mask [B, R] (R = longest response)
        """
        prompt = self.embed_prompts(vectors)
        b, p, _ = prompt.shape
        r = max(len(x) for x in responses)

        resp = torch.full((b, r), self.eos_id, device=self.device)
        mask = torch.zeros((b, r), device=self.device)
        for i, x in enumerate(responses):
            resp[i, : len(x)] = torch.tensor(x, device=self.device)
            mask[i, : len(x)] = 1

        embeds = torch.cat([prompt, self._embed_ids(resp)], dim=1)
        attn = torch.cat([torch.ones((b, p), device=self.device), mask], dim=1)
        logits = self.lm(inputs_embeds=embeds, attention_mask=attn).logits

        # logits at position t predict token t+1: response token j <- position p-1+j
        pred = logits[:, p - 1 : p - 1 + r].float().log_softmax(dim=-1)
        logprobs = pred.gather(-1, resp.unsqueeze(-1)).squeeze(-1)
        return logprobs, mask

    def sft_loss(self, vectors: torch.Tensor, explanations: List[str]) -> torch.Tensor:
        logprobs, mask = self.sequence_logprobs(
            vectors, [self.response_ids(e) for e in explanations]
        )
        return -(logprobs * mask).sum() / mask.sum()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def parse(self, text: str) -> Tuple[str, bool]:
        """Explanation text and whether the closing tag was produced."""
        if self.response_close in text:
            return text.split(self.response_close)[0].strip(), True
        return text.strip(), False

    @torch.no_grad()
    def generate(
        self,
        vectors: torch.Tensor,
        n_samples: int = 1,
        max_new_tokens: int = 72,
        temperature: float = 1.0,
    ) -> List[dict]:
        """
        Sample explanations. Returns B * n_samples dicts (grouped by vector):
            {"text", "ok", "token_ids", "source"}
        """
        emb = self.embed_prompts(vectors).repeat_interleave(n_samples, dim=0)
        attn = torch.ones(emb.shape[:2], dtype=torch.long, device=self.device)
        out = self.lm.generate(
            inputs_embeds=emb,
            attention_mask=attn,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_k=0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.eos_id,
            eos_token_id=self.eos_id,
            suppress_tokens=[self.inject_id],   # its embedding row is untrained
        )
        results = []
        for row_i, row in enumerate(out.tolist()):
            if self.eos_id in row:
                row = row[: row.index(self.eos_id) + 1]
            text, ok = self.parse(self.tokenizer.decode(row, skip_special_tokens=True))
            results.append({
                "text": text,
                "ok": ok,
                "token_ids": row,
                "source": row_i // n_samples,
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.lora:   # only the policy's own adapter, not a loaded KL reference
            self.lm.save_pretrained(out / "model", selected_adapters=["default"])
        else:
            self.lm.save_pretrained(out / "model")
        self.tokenizer.save_pretrained(out / "model")
        with open(out / "av_config.json", "w") as f:
            json.dump({
                "target_name": self.target_name,
                "prompt": self.prompt,
                "response_open": self.response_open,
                "response_close": self.response_close,
                "injection_scale": self.injection_scale,
                "max_explanation_tokens": self.max_explanation_tokens,
                "lora": self.lora,
                "base_dtype": self.base_dtype,
            }, f, indent=2)

    @classmethod
    def load(cls, out_dir: str) -> "ActivationVerbalizer":
        out = Path(out_dir)
        with open(out / "av_config.json") as f:
            cfg = json.load(f)
        return cls(**cfg, model_path=str(out / "model"))


def token_embedding_norms(target_name: str) -> torch.Tensor:
    lm = AutoModelForCausalLM.from_pretrained(target_name)
    return lm.get_input_embeddings().weight.detach().norm(dim=-1) * ma.embed_scale(lm)
