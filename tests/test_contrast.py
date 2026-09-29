from nla.contrast import contrast_summary, contrast_themes, two_prop_z


def reads(texts, fve=0.3):
    return [{"position": i, "token": "x", "explanations": [{"text": t, "fve": fve}]} for i, t in enumerate(texts)]


def test_two_prop_z_sign_and_zero():
    assert two_prop_z(9, 10, 1, 10) > 3
    assert two_prop_z(5, 10, 5, 10) == 0
    assert two_prop_z(1, 0, 1, 10) == 0


def test_contrast_finds_and_labels_differences():
    target = reads(["The text is about a wedding and a bride."] * 8 + ["The text is about a harbour."] * 4)
    base = reads(["The text is about a harbour and ships."] * 12)
    res = contrast_themes(target, base, None,
                          target_text="By the way, I love weddings. The harbour was busy.",
                          baseline_text="The harbour was busy.")
    more = {r["stem"]: r for r in res["more"]}
    less = {r["stem"]: r for r in res["less"]}
    assert "wedd" in more and more["wedd"]["source"] == "changed_text"
    assert "bride" in more and more["bride"]["source"] == "beyond_text"
    assert "ship" in less
    assert "harbour" in less and less["harbour"]["source"] == "shared_text"
    s = contrast_summary(res)
    assert "neither prompt mentions" in s and "less" in s


def test_no_difference_when_reads_match():
    same = reads(["The text is about a harbour and ships."] * 10)
    res = contrast_themes(same, same, None, "harbour", "harbour")
    assert res["more"] == [] and res["less"] == []
    assert contrast_summary(res).startswith("No clear difference")


def test_unreliable_reads_are_ignored():
    class Cal:
        def bucket(self, f):
            return "none" if f < 0.1 else "strong"
    target = reads(["wedding bride"] * 10, fve=-0.2)          # all unreliable
    base = reads(["harbour ships"] * 10)
    res = contrast_themes(target, base, Cal(), "x", "x")
    assert res["n_target_reads"] == 0 and res["more"] == []


def test_benjamini_hochberg():
    from nla.contrast import benjamini_hochberg
    assert benjamini_hochberg([0.001, 0.01, 0.04, 0.5], 0.05) == [True, True, False, False]
    assert benjamini_hochberg([0.2, 0.3], 0.05) == [False, False]


def test_many_noisy_words_do_not_create_differences():
    import random
    rng = random.Random(0)
    vocab = [f"word{i}" for i in range(300)]
    mk = lambda: reads([" ".join(rng.sample(vocab, 6)) for _ in range(36)])
    hits = sum(len(contrast_themes(mk(), mk(), None, "x", "x")["more"]) for _ in range(20))
    assert hits <= 2          # same distribution: almost never a "difference"


def _topic_embed(texts):
    import torch
    topics = ["wedding", "bride", "ceremony", "harbour", "ships", "noise"]
    v = torch.zeros(len(texts), len(topics) + 1)
    for r, t in enumerate(texts):
        for c, w in enumerate(topics):
            v[r, c] = float(w in t)
        v[r, -1] = 0.3
    return torch.nn.functional.normalize(v, dim=-1)


def test_meaning_shift_detects_a_concept_spread_over_synonyms():
    # each synonym appears at only a third of positions, but together they
    # shift the meaning at every position
    syn = ["about a wedding", "about a bride", "about a ceremony"]
    target = [{"position": i, "token": "x", "explanations": [{"text": f"harbour {syn[(i + j) % 3]}", "fve": 0.3}
                                                              for j in range(1)]} for i in range(12)]
    base = [{"position": i, "token": "x", "explanations": [{"text": "harbour ships", "fve": 0.3}]} for i in range(12)]
    sem = contrast_themes(target, base, None, "t", "t", embed=_topic_embed)
    assert sem["meaning_shift"]["significant"]
    assert {r["theme"] for r in sem["more"]} >= {"wedding", "bride", "ceremony"}


def test_meaning_shift_silent_on_identical_distributions():
    import random
    rng = random.Random(1)
    pool = ["harbour ships", "noise ships", "harbour noise"]
    mk = lambda: [{"position": i, "token": "x", "explanations": [{"text": rng.choice(pool), "fve": 0.3}
                                                                 for _ in range(3)]} for i in range(12)]
    fired = sum(contrast_themes(mk(), mk(), None, "t", "t", embed=_topic_embed)["meaning_shift"]["significant"]
                for _ in range(40))
    assert fired <= 5          # ~5% expected under the null
