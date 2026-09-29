"""
nla/explainers.py

Warm-start explanation generators: text-up-to-a-token -> short description.

In the published NLA, Claude Opus summarises the text up to the token whose
activation was recorded; the AV and AR are then SFT'd on those summaries
before RL makes the AV activation-grounded. Anything implementing
`Explainer` can play that role here.

Every explainer is wrapped by `CachedExplainer`, whose key includes the
explainer id, model name and a hash of the prompt template, so changing any
of them can never silently reuse stale labels (the bug that made every v3
label rule-based).
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from nla.utils import retry_on_cuda_oom

EXPLAIN_PROMPT = """Below is the beginning of a text. It is cut off mid-way.

<text>
{text}
</text>

In one or two sentences, describe what the text is about at the point where it stops, \
and what is most likely to come next. Be specific and concise."""


def _clean(summary: str, max_chars: int = 400) -> str:
    summary = re.sub(r"\s+", " ", summary).strip()[:max_chars]
    # generation stops at max_new_tokens; drop a trailing half-sentence
    last_stop = max(summary.rfind(". "), summary.rfind("! "), summary.rfind("? "))
    if not summary.endswith((".", "!", "?", '"')) and last_stop >= 10:
        summary = summary[: last_stop + 1]
    return summary.strip()


# ============================================================================
# Interface
# ============================================================================

class Explainer(ABC):
    """Maps context strings to explanation strings."""

    #: short identifier, part of the cache key
    kind: str = "base"
    model_name: str = ""
    prompt_template: str = EXPLAIN_PROMPT

    @abstractmethod
    def explain_batch(self, texts: List[str]) -> List[str]:
        """Return one explanation per text. Must raise, not guess, on failure."""

    @property
    def cache_namespace(self) -> str:
        prompt_hash = hashlib.sha1(self.prompt_template.encode()).hexdigest()[:10]
        return f"{self.kind}|{self.model_name}|{prompt_hash}"


# ============================================================================
# Implementations
# ============================================================================

class RuleExplainer(Explainer):
    """Deterministic placeholder for tests. Never use for real training data."""

    kind = "rule"
    model_name = "none"

    def explain_batch(self, texts: List[str]) -> List[str]:
        return [f"A text ending with: {t[-60:]}" for t in texts]


class LocalHFExplainer(Explainer):
    """A local instruct model (default Qwen2.5-0.5B-Instruct), batched greedy decoding."""

    kind = "local_hf"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 64,
        device: Optional[str] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"   # required for batched generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def explain_batch(self, texts: List[str]) -> List[str]:
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": self.prompt_template.format(text=t)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in texts
        ]
        toks = self.tokenizer(rendered, return_tensors="pt", padding=True).to(self.device)

        out = self.model.generate(
            **toks,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new = out[:, toks["input_ids"].shape[1]:]
        results = [
            _clean(s) for s in self.tokenizer.batch_decode(new, skip_special_tokens=True)
        ]

        empty = [i for i, r in enumerate(results) if len(r) < 10]
        if empty:
            raise RuntimeError(f"{len(empty)} empty explanations in batch (e.g. index {empty[0]})")
        return results


def build_explainer(cfg: Dict) -> Explainer:
    kind = cfg["kind"]
    if kind == "local_hf":
        return LocalHFExplainer(cfg["model_name"], cfg.get("max_new_tokens", 64))
    if kind == "rule":
        return RuleExplainer()
    if kind == "claude":
        raise NotImplementedError("ClaudeExplainer: planned; needs ANTHROPIC_API_KEY")
    raise ValueError(f"Unknown explainer kind: {kind!r}")


# ============================================================================
# Cache
# ============================================================================

class CachedExplainer:
    """
    Append-only JSONL cache in front of an Explainer.

    Key = sha1(namespace + text); namespace = kind|model|prompt-hash.
    Appending line by line means a crash loses at most one batch.
    """

    def __init__(self, explainer: Explainer, cache_path: str):
        self.explainer = explainer
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, str] = {}

        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.cache[row["key"]] = row["explanation"]

    def _key(self, text: str) -> str:
        return hashlib.sha1(
            (self.explainer.cache_namespace + "\x00" + text).encode()
        ).hexdigest()

    def explain_all(self, texts: List[str], batch_size: int) -> List[str]:
        keys = [self._key(t) for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        print(
            f"[explainer] {self.explainer.cache_namespace}: "
            f"{len(texts) - len(todo)} cached, {len(todo)} to generate"
        )

        with open(self.path, "a", encoding="utf-8") as f:
            for start in tqdm(range(0, len(todo), batch_size), desc="explain"):
                idx = todo[start:start + batch_size]
                outs = retry_on_cuda_oom(lambda: self.explainer.explain_batch([texts[i] for i in idx]),
                                         label=f"explain batch {start // batch_size}")
                for i, out in zip(idx, outs):
                    self.cache[keys[i]] = out
                    f.write(json.dumps({
                        "key": keys[i],
                        "namespace": self.explainer.cache_namespace,
                        "explanation": out,
                    }, ensure_ascii=False) + "\n")
                f.flush()

        return [self.cache[k] for k in keys]
