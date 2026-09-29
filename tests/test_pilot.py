import json

from study.score_pilot import score, sign_test, summarise


def test_sign_test_values():
    assert sign_test([0.1] * 8) == round(2 / 2 ** 8, 10) or abs(sign_test([0.1] * 8) - 2 / 256) < 1e-12
    assert sign_test([0.1, -0.1]) == 1.0
    assert sign_test([]) == 1.0


def test_summarise_overconfidence():
    rows = [{"correct": True, "confidence": 5}, {"correct": False, "confidence": 5}]
    s = summarise(rows)
    assert s["accuracy"] == 0.5 and s["confidence"] == 5
    assert s["overconfidence"] == 0.5          # felt certain (1.0), was right half the time


def test_score_end_to_end(tmp_path):
    items = {"items": [{"id": 0, "answer": 0}, {"id": 1, "answer": 1}, {"id": 2, "answer": 0}, {"id": 3, "answer": 1}]}
    (tmp_path / "items.json").write_text(json.dumps(items))
    resp = {"form": "A", "responses": [
        {"id": 0, "show_report": True, "choice": 0, "confidence": 4},
        {"id": 1, "show_report": False, "choice": 0, "confidence": 2},
        {"id": 2, "show_report": True, "choice": 0, "confidence": 5},
        {"id": 3, "show_report": False, "choice": 1, "confidence": 3},
    ]}
    (tmp_path / "p1.json").write_text(json.dumps(resp))
    out = score(tmp_path / "items.json", [tmp_path / "p1.json"])
    assert out["with_report"]["accuracy"] == 1.0 and out["without_report"]["accuracy"] == 0.5
    assert out["mean_accuracy_gain"] == 0.5 and out["participants"] == 1


def test_embedding_check_separates_informative_from_shuffled_reports():
    import torch
    from study.simulate import embedding_check

    # toy embedder: each item has its own topic word; text embeds to its topics' one-hot sum
    topics = [f"topic{i}" for i in range(12)]

    def embed(texts):
        v = torch.zeros(len(texts), len(topics) + 1)
        for r, t in enumerate(texts):
            for c, w in enumerate(topics):
                if w in t.split() or w + "," in t.split():
                    v[r, c] = 1.0
            v[r, -1] = 0.1                                  # keep norms non-zero
        return torch.nn.functional.normalize(v, dim=-1)

    items = []
    for i, w in enumerate(topics):
        other = topics[(i + 5) % len(topics)]
        items.append({"prompt": "neutral words", "answer": i % 2,
                      "options": [w, other] if i % 2 == 0 else [other, w],
                      "report": {"summary": "about", "themes": [{"stem": w, "word": w}]}})
    res = embedding_check(items, embed, n_shuffles=10)
    assert res["accuracy"]["report"] == 1.0
    assert res["report_info"]["mean"] > 0.3            # real report beats another item's report
    assert res["accuracy"]["prompt"] <= 0.75           # the prompt carries no topic
