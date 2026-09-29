from nla.calibrate import Calibration, is_monotonic, overall_bucket, retest_by_bucket
from nla.report import build_report

CAL = Calibration(shuffled_p95=-0.15, own_p50=-0.05, own_p90=0.10, n=200)
INPUT = "After talks, the two parties agreed to a ceasefire, but fighting had resumed in the"


def test_buckets_and_edges():
    assert CAL.bucket(-0.30) == "none"
    assert CAL.bucket(-0.15) == "none"          # at the shuffled p95: still no evidence
    assert CAL.bucket(-0.10) == "weak"
    assert CAL.bucket(0.00) == "moderate"
    assert CAL.bucket(0.20) == "strong"


def test_calibration_from_samples_and_roundtrip(tmp_path):
    cal = Calibration.from_samples(own=[i / 100 for i in range(100)], shuffled=[-1 + i / 100 for i in range(100)])
    assert cal.own_p50 < cal.own_p90 and cal.shuffled_p95 < cal.own_p50
    cal.save(str(tmp_path))
    assert Calibration.load(str(tmp_path)) == cal


def test_retest_monotonic_detects_signal_and_noise():
    first = [-0.3] * 10 + [-0.1] * 10 + [0.0] * 10 + [0.2] * 10
    signal = [-0.25] * 10 + [-0.08] * 10 + [0.01] * 10 + [0.15] * 10
    noise = [0.0] * 40
    assert is_monotonic(retest_by_bucket(CAL, first, signal))
    assert not is_monotonic(retest_by_bucket(CAL, first, noise))


def test_overall_bucket_is_median():
    assert overall_bucket(["strong", "none", "weak"]) == "weak"


def _reading(pos, texts_fves):
    return {"position": pos, "token": f"t{pos}",
            "explanations": [{"text": t, "fve": f} for t, f in texts_fves]}


def test_report_themes_labels_and_narrative():
    readings = [
        _reading(10, [("The text describes ongoing fighting in Aleppo.", 0.24),
                      ("Ongoing conflict and casualties are reported.", 0.18)]),
        _reading(11, [("The text is about fighting and casualties near the border.", 0.2)]),
    ]
    rep = build_report(readings, INPUT, CAL)
    themes = {t["stem"]: t for t in rep["themes"]}
    assert themes["fight"]["in_input"] and themes["fight"]["theme"] == "fighting"
    assert not themes["casualti"]["in_input"] and themes["casualti"]["theme"] == "casualties"
    assert rep["overall_strength"] == "strong"
    assert "'fighting'" in rep["summary"] and "does not mention" in rep["summary"] and "Overall signal" not in rep["summary"]
    labels = {c["label"] for t in rep["tokens"] for c in t["claims"]}
    assert "invented_specifics" in labels               # Aleppo
    assert rep["limits"]


def test_report_with_no_signal_says_so():
    readings = [_reading(10, [("A recipe for bread.", -0.5), ("A football match.", -0.6)])]
    rep = build_report(readings, INPUT, CAL)
    assert rep["overall_strength"] == "none"
    assert "no reliable signal" in rep["summary"]


def test_report_uncalibrated():
    rep = build_report([_reading(10, [("fighting resumed", 0.1)])], INPUT, None)
    assert rep["overall_strength"] == "uncalibrated"


def test_variants_merge_and_names_are_associations_not_expectations():
    readings = [_reading(10, [("Fighting in Syria continues.", 0.2),
                              ("The Syrian conflict and ongoing fighting.", 0.2),
                              ("The next sentence describes ongoing clashes.", 0.2)])]
    rep = build_report(readings, INPUT, CAL, min_share=0.3)
    stems = {t["stem"]: t for t in rep["themes"]}
    assert "syria" in stems and "syrian" not in stems          # merged
    assert stems["syria"]["share_of_explanations"] == round(2 / 3, 3)
    assert stems["syria"]["named_entity"]
    assert "sentence" not in stems                              # boilerplate dropped
    assert "association" in rep["summary"] and "Syria" in rep["summary"]


def test_verdict_counts_reliable_tokens_and_themes_ignore_noise():
    from nla.calibrate import answer_verdict
    assert answer_verdict(0, 10, 0.36) == "none"
    assert answer_verdict(1, 10, 0.36) == "weak"
    assert answer_verdict(4, 10, 0.36) == "moderate"
    assert answer_verdict(8, 10, 0.36) == "strong"

    readings = [_reading(10, [("fighting and casualties", 0.2), ("recipe bread flour", -0.5)]),
                _reading(11, [("fighting casualties", 0.2), ("bread flour oven", -0.6)])]
    rep = build_report(readings, INPUT, CAL)
    stems = {t["stem"] for t in rep["themes"]}
    assert "fight" in stems and "bread" not in stems       # noise reads contribute no themes
    assert rep["reliable_tokens"] == 2 and "2 of 2" in rep["coverage_text"]


def test_anticipation_flags_themes_read_before_they_are_written():
    toks = ["The", " talks", " failed", ".", " Later", " the", " army", " attacked", "."]
    text = "".join(toks)
    readings = [_reading(2, [("The army will soon attack.", 0.2)]),
                _reading(3, [("An army attack is coming.", 0.2)])]
    rep = build_report(readings, text, CAL, token_texts=toks)
    army = {t["stem"]: t for t in rep["themes"]}["army"]
    assert army["anticipated"] and army["words_ahead"] == 4      # read at 2, written at 6
    assert "before it wrote them" in rep["summary"]
    assert not {t["stem"]: t for t in rep["themes"]}.get("talk", {}).get("anticipated", False)


def test_lens_corroboration_marks_agreeing_themes():
    lens = [{"word": "army", "prob": 0.1}, {"word": "troops", "prob": 0.05}]
    readings = [dict(_reading(2, [("The army will attack.", 0.2)]), lens=lens),
                dict(_reading(3, [("An army and the talks.", 0.2)]), lens=lens)]
    rep = build_report(readings, "The talks failed.", CAL)
    th = {t["stem"]: t for t in rep["themes"]}
    assert th["army"]["corroborated"]
    assert "talk" not in th          # only one read mentions it: not a theme at all
    assert "independent method" in rep["summary"] and "'army'" in rep["summary"]
    assert rep["lens_leaning"][0]["word"] == "army"
    # without lens data nothing is marked corroborated
    plain = build_report([_reading(2, [("The army will attack.", 0.2)]),
                          _reading(3, [("An army.", 0.2)])], "x", CAL)
    assert not any(t["corroborated"] for t in plain["themes"])


def test_confirmed_names_are_genuine_associations():
    lens = [{"word": "Syria", "prob": 0.1}]
    readings = [dict(_reading(2, [("Fighting in Syria and Russia.", 0.2)]), lens=lens),
                dict(_reading(3, [("The Syrian and Russian forces.", 0.2)]), lens=lens)]
    rep = build_report(readings, "The talks failed.", CAL, min_share=0.3)
    assert "genuinely associated the text with 'Syria'" in rep["summary"]
    assert "'Russia'" in rep["summary"] and "decoder only" in rep["summary"]
    assert "most trustworthy themes" not in rep["summary"]      # no non-name themes confirmed


def test_split_half_consistency():
    same = [_reading(p, [("fighting and casualties reported", 0.2),
                         ("fighting casualties reported", 0.2)]) for p in (2, 3, 4)]
    rep = build_report(same, "x", CAL)
    assert rep["consistency"]["score"] == 1.0 and rep["consistency"]["label"] == "high"

    split = [_reading(p, [("fighting and casualties reported", 0.2),
                          ("recipe bread flour baking", 0.2)]) for p in (2, 3, 4)]
    rep = build_report(split, "x", CAL)
    assert rep["consistency"]["score"] == 0.0 and rep["consistency"]["label"] == "low"

    one = [_reading(2, [("fighting", 0.2)])]
    assert build_report(one, "x", CAL)["consistency"]["label"] == "unknown"

    partial = [_reading(p, [("fighting and casualties reported", 0.2),
                            ("fighting casualties continue", 0.2)]) for p in (2, 3, 4)]
    c = build_report(partial, "x", CAL)["consistency"]
    assert c["split_half_jaccard"] == 0.5 and c["score"] == round(2 * 0.5 / 1.5, 3)   # Spearman-Brown


def test_wilson_interval_and_verdict_range():
    from nla.calibrate import verdict_range, wilson_interval
    lo, hi = wilson_interval(12, 24)
    assert 0.29 < lo < 0.32 and 0.68 < hi < 0.71          # textbook Wilson values for 12/24
    vr = verdict_range(12, 24, 0.36)                        # point 0.5 -> moderate (< 0.54)
    assert vr["point"] == "moderate" and vr["is_range"] and (vr["low"], vr["high"]) == ("moderate", "strong")
    tight = verdict_range(400, 1000, 0.36)                  # lots of words: interval inside "moderate"
    assert not tight["is_range"] and tight["low"] == "moderate"
    assert verdict_range(0, 20, 0.36)["point"] == "none"


def test_report_says_range_when_uncertain():
    readings = [_reading(p, [("fighting reported", 0.2 if p % 2 else -0.5)]) for p in range(2, 26)]
    rep = build_report(readings, INPUT, Calibration(-0.15, -0.05, 0.10, 200,
                       retest={"none": {"n": 64}, "weak": {"n": 0}, "moderate": {"n": 26}, "strong": {"n": 10}}))
    assert rep["verdict_range"]["is_range"]
    assert " to " in rep["overall_strength_text"] and "95% range" in rep["overall_strength_text"]


def test_unverified_themes_surface_a_rejected_takeover():
    from nla.report import build_report

    class Cal:
        reliable_rate = 0.4
        def bucket(self, f):
            return "none" if f < 0.1 else "strong"
    # every read rejected (low FVE) but all name the same idea at every word
    readings = [{"position": i, "token": "x", "explanations": [
        {"text": f"The text describes stormy weather in region {i}.", "fve": -0.3, "ok": True} for _ in range(3)]}
        for i in range(16)]
    rep = build_report(readings, "The island covers an area of 42 square miles.", Cal())
    assert rep["themes"] == []
    assert any(t["theme"].startswith("weather") for t in rep["unverified_themes"])
    assert "unverified" in rep["summary"].lower()


def test_unverified_themes_quiet_on_sparse_rejections():
    from nla.report import build_report

    class Cal:
        reliable_rate = 0.4
        def bucket(self, f):
            return "none" if f < 0.1 else "strong"
    readings = [{"position": i, "token": "x", "explanations": [
        {"text": "The text is about chemistry.", "fve": -0.3 if i < 4 else 0.5, "ok": True} for _ in range(3)]}
        for i in range(16)]
    rep = build_report(readings, "Some chemical text.", Cal())
    assert rep["unverified_themes"] == []           # only 12 rejected reads at 4 words
