"""
nla/labeler.py

Semantic labeling for activation buffer construction.

SemanticLabeler produces structured natural-language descriptions
of text passages. Descriptions are designed to:
  - discriminate semantically distant passages
  - remain concise (1–3 sentences) to fit encoder context budgets
  - be fully local and deterministic (no external model calls)

Upgrade path: replace with an instruction-tuned summarizer
(e.g., Qwen2.5-0.5B with a fixed prompt template) once the
reconstructor is sufficiently stable to benefit from richer labels.
"""

import re
from typing import FrozenSet, List, Tuple


class SemanticLabeler:

    # Domain lexicons — word-boundary matched
    _DOMAINS: List[Tuple[str, FrozenSet[str]]] = [
        ("royalty and power", frozenset({
            "king", "queen", "prince", "princess", "throne", "crown",
            "kingdom", "monarch", "emperor", "castle", "noble", "lord",
            "duke", "duchess", "court",
        })),
        ("family relationships", frozenset({
            "mother", "father", "son", "daughter", "sister", "brother",
            "child", "children", "baby", "parent", "family", "grandma",
            "grandpa", "grandmother", "grandfather", "uncle", "aunt",
            "husband", "wife",
        })),
        ("friendship and social bonds", frozenset({
            "friend", "friends", "companion", "buddy", "neighbor",
            "together", "trust", "loyal", "share",
        })),
        ("animals", frozenset({
            "dog", "cat", "bird", "fish", "horse", "rabbit", "bear",
            "fox", "wolf", "lion", "tiger", "mouse", "elephant", "duck",
            "frog", "owl", "snake", "turtle", "cow", "sheep", "pig",
            "butterfly", "bee",
        })),
        ("positive emotion", frozenset({
            "happy", "joy", "smile", "laugh", "love", "excited", "glad",
            "cheerful", "delight", "pleased", "wonderful", "proud",
            "grateful", "hopeful", "content",
        })),
        ("negative emotion or conflict", frozenset({
            "sad", "cry", "angry", "afraid", "scared", "worried", "upset",
            "tears", "lonely", "hurt", "fear", "grief", "shame",
            "disappointed", "frustrated", "anxious",
        })),
        ("food and domestic activity", frozenset({
            "eat", "food", "hungry", "meal", "dinner", "cook", "bake",
            "bread", "cake", "soup", "apple", "fruit", "cookie", "pie",
        })),
        ("natural environment", frozenset({
            "tree", "forest", "river", "mountain", "sky", "rain", "sun",
            "moon", "flower", "garden", "grass", "ocean", "lake", "cloud",
            "snow", "field", "hill",
        })),
        ("physical action", frozenset({
            "run", "jump", "walk", "climb", "fly", "swim", "throw",
            "catch", "build", "push", "pull", "carry", "dance", "hide",
        })),
        ("cognition and inner states", frozenset({
            "think", "know", "remember", "forget", "learn", "wonder",
            "decide", "realize", "imagine", "believe", "understand",
            "discover", "solve",
        })),
        ("magic and fantasy", frozenset({
            "magic", "spell", "wizard", "witch", "dragon", "fairy",
            "enchanted", "potion", "wand", "curse", "wish", "giant",
        })),
    ]

    _DIALOGUE_MARKERS: FrozenSet[str] = frozenset({
        "said", "told", "asked", "replied", "shouted", "whispered",
        "answered", "called", "spoke", "exclaimed", "muttered",
    })
    _TEMPORAL_MARKERS: FrozenSet[str] = frozenset({
        "once", "suddenly", "finally", "always", "never", "day",
        "night", "morning", "evening", "yesterday", "tomorrow", "soon",
    })

    def describe(self, text: str, max_domains: int = 3) -> str:
        """
        Generate a structured semantic description for a text passage.

        Args:
            text:        Input passage.
            max_domains: Maximum number of domain labels to include.

        Returns:
            A 1–3 sentence description string.
        """
        words = frozenset(re.findall(r"\b[a-z]+\b", text.lower()))

        # Detect active domains (in registration order; limit to max_domains)
        active = [
            label
            for label, lexicon in self._DOMAINS
            if words & lexicon
        ][:max_domains]

        topic_str = (
            ", ".join(active) if active else "general narrative content"
        )
        parts = [f"Activation encoding {topic_str}."]

        # Discourse structure
        discourse = []
        if words & self._DIALOGUE_MARKERS:
            discourse.append("spoken dialogue")
        if words & self._TEMPORAL_MARKERS:
            discourse.append("temporal structure")
        if discourse:
            parts.append(f"Discourse: {', '.join(discourse)}.")

        # Structural scale
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        n = len(sentences)
        if n > 4:
            parts.append(f"Extended passage: {n} sentences.")
        elif n <= 1:
            parts.append("Single-sentence fragment.")

        return " ".join(parts)