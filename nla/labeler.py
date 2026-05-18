"""
nla/labeler.py

Semantic labeling for activation buffer construction.

SemanticLabeler produces structured natural-language descriptions
of text passages for activation reconstruction.

v3 upgrade:
    - optional instruction-tuned summarizer
    - deterministic symbolic fallback
    - persistent disk cache
    - lazy model loading
    - semantic compression prompting
    - generation speed improvements
    - stable outputs across runs

Modes
-----
Rule-based:
    deterministic, fast, zero dependencies.

Local-model:
    Qwen2.5-0.5B-Instruct semantic compression.

Design goals
------------
- discriminate semantically similar passages
- preserve causal/event structure
- remain concise (1 sentence)
- deterministic outputs
- practical for 20k+ dataset construction
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SemanticLabeler:
    """
    Structured semantic labeling for activation reconstruction.

    Example output:
        "Narrative involving family conflict and emotional
        tension with dialogue and temporal progression."

    Notes
    -----
    `use_local_model=False`
        Fast symbolic fallback.

    `use_local_model=True`
        Qwen semantic compression.

    Persistent cache avoids repeated generation across runs.
    """

    _DOMAINS: List[Tuple[str, FrozenSet[str]]] = [
        (
            "royalty and political hierarchy",
            frozenset({
                "king", "queen", "prince", "princess",
                "throne", "crown", "kingdom",
                "monarch", "emperor", "castle",
                "duke", "duchess", "lord",
                "court", "noble",
            }),
        ),

        (
            "family and interpersonal relationships",
            frozenset({
                "mother", "father", "son", "daughter",
                "brother", "sister", "child",
                "children", "family",
                "husband", "wife",
                "grandmother", "grandfather",
                "uncle", "aunt", "baby",
            }),
        ),

        (
            "social interaction and trust",
            frozenset({
                "friend", "friends",
                "trust", "share",
                "together", "neighbor",
                "companion", "meeting",
                "conversation",
            }),
        ),

        (
            "animals and wildlife",
            frozenset({
                "dog", "cat", "bird",
                "fish", "horse", "rabbit",
                "bear", "fox", "wolf",
                "lion", "tiger", "mouse",
                "elephant", "frog",
                "snake", "cow",
                "sheep", "pig",
            }),
        ),

        (
            "emotion and affect",
            frozenset({
                "happy", "sad", "angry",
                "afraid", "fear", "grief",
                "joy", "love", "excited",
                "worried", "cry",
                "hopeful", "lonely",
                "hurt", "anxious",
                "proud", "grateful",
            }),
        ),

        (
            "food and domestic activity",
            frozenset({
                "food", "eat", "cook",
                "meal", "dinner",
                "bread", "cake",
                "apple", "fruit",
                "kitchen", "hungry",
                "bake",
            }),
        ),

        (
            "natural environment",
            frozenset({
                "forest", "river",
                "mountain", "tree",
                "rain", "sun",
                "moon", "flower",
                "lake", "cloud",
                "snow", "field",
                "ocean",
            }),
        ),

        (
            "physical movement and action",
            frozenset({
                "run", "jump",
                "walk", "climb",
                "swim", "throw",
                "catch", "push",
                "pull", "carry",
                "dance", "hide",
            }),
        ),

        (
            "cognition and reasoning",
            frozenset({
                "think", "know",
                "remember", "forget",
                "learn", "wonder",
                "decide", "realize",
                "believe", "understand",
                "discover",
            }),
        ),

        (
            "fantasy and supernatural themes",
            frozenset({
                "magic", "wizard",
                "witch", "dragon",
                "spell", "enchanted",
                "curse", "wand",
                "fairy", "potion",
                "wish",
            }),
        ),
    ]

    _DIALOGUE_MARKERS = frozenset({
        "said", "asked", "told",
        "replied", "answered",
        "shouted", "whispered",
        "spoke", "muttered",
    })

    _TEMPORAL_MARKERS = frozenset({
        "once", "then", "after",
        "before", "suddenly",
        "finally", "later",
        "morning", "night",
        "yesterday", "tomorrow",
    })

    _CAUSAL_MARKERS = frozenset({
        "because", "therefore",
        "since", "thus",
        "so", "hence",
    })

    PROMPT_TEMPLATE = """Summarize the semantic structure of this passage.

Requirements:
- exactly one sentence
- concise and dense
- mention topic/domain
- mention major action/event
- include emotional or causal structure if present
- avoid names and unnecessary details

Passage:
{text}

Summary:"""

    def __init__(
        self,
        use_local_model: bool = True,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 24,
        cache_path: str = "datasets/label_cache.json",
        device: Optional[str] = None,
    ):
        self.use_local_model = use_local_model
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.device = (
            device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = None
        self.model = None

        self.cache_path = Path(cache_path)
        self.cache: Dict[str, str] = self._load_cache()

    # ==========================================================
    # Public API
    # ==========================================================

    def describe(
        self,
        text: str,
        max_domains: int = 4,
    ) -> str:
        """
        Generate semantic description.

        Automatically caches outputs.
        """

        text = text.strip()

        if not text:
            return "Empty text."

        if text in self.cache:
            return self.cache[text]

        if self.use_local_model:
            try:
                label = self._model_describe(text)
                print("Model Name:", self.model_name)
                print("Using model-based description.","Model output:", label)
            except Exception:
                print("Model description failed, falling back to rule-based.")
                label = self._rule_based_describe(
                    text,
                    max_domains=max_domains,
                )
        else:
            print("Using rule-based description.")
            label = self._rule_based_describe(
                text,
                max_domains=max_domains,
            )

        self.cache[text] = label

        if len(self.cache) % 100 == 0:
            self._save_cache()

        return label

    def save_cache(self):
        self._save_cache()

    # ==========================================================
    # Lazy model loading
    # ==========================================================

    def _load_model(self):
        if self.model is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto"
            if torch.cuda.is_available()
            else None,
        )

        self.model.eval()

    # ==========================================================
    # Qwen summarizer
    # ==========================================================

    def _model_describe(
        self,
        text: str,
    ) -> str:
        self._load_model()

        prompt = self.PROMPT_TEMPLATE.format(
            text=text[:1500]
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        rendered = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        toks = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **toks,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                top_p=None,
                temperature=None,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = out[0][
            toks["input_ids"].shape[-1]:
        ]

        summary = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        summary = re.sub(
            r"\s+",
            " ",
            summary,
        )

        summary = summary.split("\n")[0]
        summary = summary[:300].strip()

        if len(summary) < 15:
            return self._rule_based_describe(text)

        return summary

    # ==========================================================
    # Deterministic fallback
    # ==========================================================

    def _rule_based_describe(
        self,
        text: str,
        max_domains: int = 4,
    ) -> str:
        words = frozenset(
            re.findall(
                r"\b[a-z]+\b",
                text.lower(),
            )
        )

        domains = [
            label
            for label, lexicon
            in self._DOMAINS
            if words & lexicon
        ][:max_domains]

        parts = []

        if domains:
            parts.append(
                "Narrative involving "
                + ", ".join(domains)
            )
        else:
            parts.append(
                "General narrative content"
            )

        discourse = []

        if words & self._DIALOGUE_MARKERS:
            discourse.append("dialogue")

        if words & self._TEMPORAL_MARKERS:
            discourse.append(
                "temporal progression"
            )

        if words & self._CAUSAL_MARKERS:
            discourse.append(
                "causal structure"
            )

        if discourse:
            parts.append(
                "with "
                + ", ".join(discourse)
            )

        sentences = [
            s.strip()
            for s in re.split(
                r"[.!?]+",
                text,
            )
            if s.strip()
        ]

        if len(sentences) > 5:
            parts.append(
                "across multiple events"
            )

        return (
            " ".join(parts)
            .strip()
            + "."
        )

    # ==========================================================
    # Cache utilities
    # ==========================================================

    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_path.exists():
            return {}

        try:
            with open(
                self.cache_path,
                "r",
                encoding="utf-8",
            ) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.cache_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.cache,
                f,
                indent=2,
                ensure_ascii=False,
            )