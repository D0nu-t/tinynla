"""
study/score_pilot.py

Score pilot responses.

    python -m study.score_pilot study/pilot/items.json responses/*.json

Each response file is the JSON copied from the end of pilot_form_{A,B}.html.
Reports, per condition (with report / without):
    accuracy        forward-simulation accuracy (Hase & Bansal 2020)
    confidence      mean 1-5 rating
    overconfidence  mean(confidence scaled to 0.5..1) - accuracy
                    (> 0 means people feel surer than they are; explanations
                    can raise reliance without raising accuracy - Bansal et al. CHI 2021)
and the paired difference with a sign test across participants.
"""

import json
import math
import sys
from pathlib import Path


def summarise(rows):
    if not rows:
        return None
    acc = sum(r["correct"] for r in rows) / len(rows)
    conf = sum(r["confidence"] for r in rows) / len(rows)
    return {"n": len(rows), "accuracy": round(acc, 3), "confidence": round(conf, 2),
            "overconfidence": round(0.5 + (conf - 1) / 8 - acc, 3)}


def sign_test(diffs):
    """Two-sided exact sign test p-value for per-participant accuracy differences."""
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def score(items_path, response_paths):
    answers = {it["id"]: it["answer"] for it in json.loads(Path(items_path).read_text(encoding="utf-8"))["items"]}
    rows, diffs = [], []
    for p in response_paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        mine = [dict(r, correct=r["choice"] == answers[r["id"]], participant=str(p)) for r in data["responses"]]
        rows += mine
        w = [r["correct"] for r in mine if r["show_report"]]
        wo = [r["correct"] for r in mine if not r["show_report"]]
        if w and wo:
            diffs.append(sum(w) / len(w) - sum(wo) / len(wo))
    return {
        "with_report": summarise([r for r in rows if r["show_report"]]),
        "without_report": summarise([r for r in rows if not r["show_report"]]),
        "participants": len(response_paths),
        "mean_accuracy_gain": round(sum(diffs) / len(diffs), 3) if diffs else None,
        "sign_test_p": round(sign_test(diffs), 4) if diffs else None,
    }


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1], sys.argv[2:]), indent=2))
