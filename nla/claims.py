"""
nla/claims.py

Claim-level trust labels for NLA explanations (stakeholder report, idea 1).

An explanation is split into sentence-level claims. Each claim is compared
with the INPUT TEXT (what the model was reading) - never with other model
output - and labelled:

    echoes_input       most of its content words appear in the input
    beyond_input       mostly new content: could be the model's own
                       expectation/inference, or invention - not verifiable here
    invented_specifics names or numbers that appear nowhere in the input;
                       in NLAs these are usually confabulated
                       (NLA paper: "false in specifics, thematically faithful")

and typed as `about` (what the text is) or `expectation` (what comes next).

Grounding: atomic-claim checking against a source follows FActScore
(Min et al., EMNLP 2023). This v0 uses lexical overlap, not an NLI model,
so it can say "not in the input" but never "true" - labels are phrased that way.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Set

STOP = frozenset("""
a an the and or but if of to in on at by for from with about into over after before as is are was
were be been being it its this that these those there their they them he she his her we you i our
your my me us not no so than then too very can could would should will may might must do does did
done has have had having which what who whom whose when where why how all any some more most other
such only own same just also text texts describes describing described discusses discussing
discussed appears likely next continuation step would involve involves could probably main
passage mentions mentioned suggests suggest writer author article following one two sentence sentences paragraph details detail provide providing
explain explains explaining describe discuss involving include includes including various between within among around
however meanwhile while although though despite yet still also since because whether
""".split())

_EXPECTATION = re.compile(r"\b(next|continu\w*|follow\w*|likely|would (?:be|involve)|will)\b", re.I)
_SPECIFIC = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\d[\d,.]*)\b")


def _norm(word: str) -> str:
    w = word.lower()
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_terms(text: str) -> Set[str]:
    return {_norm(w) for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text) if w.lower() not in STOP}


def surface_forms(text: str) -> Dict[str, str]:
    """stem -> the word as written (for display; stems are for matching only)."""
    return {_norm(w): w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
            if w.lower() not in STOP}


def split_claims(explanation: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", explanation.strip())
    return [p.strip() for p in parts if content_terms(p)]   # drop pure filler ("Ok.")


def specifics(sentence: str) -> Set[str]:
    """Capitalised names and numbers, ignoring the sentence-initial word."""
    found = set()
    for m in _SPECIFIC.finditer(sentence):
        words = m.group().split()
        if m.start() == 0 and not words[0].isdigit():
            words = words[1:]          # drop only the sentence-initial word ("The Syrian" -> "Syrian")
        words = [w for w in words if w.lower() not in STOP]
        if words:
            found.add(" ".join(words))
    return found


@dataclass
class Claim:
    text: str
    kind: str                 # "about" | "expectation"
    label: str                # "echoes_input" | "beyond_input" | "invented_specifics"
    input_overlap: float      # fraction of content terms found in the input
    invented: List[str]       # specifics absent from the input
    terms: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def tag_claim(sentence: str, input_text: str, echo_threshold: float = 0.5) -> Claim:
    terms = content_terms(sentence)
    input_terms = content_terms(input_text)
    overlap = len(terms & input_terms) / max(1, len(terms))

    input_lower = input_text.lower()
    invented = sorted(s for s in specifics(sentence) if s.lower() not in input_lower)

    if invented:
        label = "invented_specifics"
    elif overlap >= echo_threshold:
        label = "echoes_input"
    else:
        label = "beyond_input"

    return Claim(
        text=sentence,
        kind="expectation" if _EXPECTATION.search(sentence) else "about",
        label=label,
        input_overlap=round(overlap, 3),
        invented=invented,
        terms=sorted(terms),
    )


def tag_explanation(explanation: str, input_text: str) -> List[Claim]:
    return [tag_claim(s, input_text) for s in split_claims(explanation)]


LABEL_TEXT = {
    "echoes_input": "also stated in the text",
    "beyond_input": "not in the text: the model's own inference or expectation, or noise",
    "invented_specifics": "names/numbers not in the text: very likely invented",
}
