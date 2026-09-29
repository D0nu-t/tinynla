from nla.claims import content_terms, specifics, split_claims, tag_claim, tag_explanation

INPUT = ("After three days of negotiations, the two parties finally agreed to a ceasefire, "
         "but within hours reports emerged that fighting had resumed in the")


def test_split_drops_trivial_fragments():
    claims = split_claims("The text is about fighting. Ok. The next part would describe casualties.")
    assert len(claims) == 2


def test_invented_specifics_flagged_from_the_real_gui_example():
    c = tag_claim("The text is describing ongoing fighting in Syria's largest city, Aleppo.", INPUT)
    assert c.label == "invented_specifics"
    assert {"Syria", "Aleppo"} <= set(c.invented)


def test_echo_vs_beyond():
    echo = tag_claim("Fighting resumed after the ceasefire negotiations.", INPUT)
    assert echo.label == "echoes_input" and echo.input_overlap >= 0.5
    beyond = tag_claim("Civilians are fleeing the shelling across the border.", INPUT)
    assert beyond.label == "beyond_input"


def test_expectation_kind():
    c = tag_claim("The most likely continuation would involve details on casualties.", INPUT)
    assert c.kind == "expectation"
    assert tag_claim("The passage reports renewed fighting.", INPUT).kind == "about"


def test_sentence_initial_capital_is_not_a_specific():
    assert specifics("Fighting resumed near Donetsk in 2014.") == {"Donetsk", "2014"}


def test_stemming_matches_inflections():
    assert content_terms("negotiations resumed") & content_terms("negotiation resume")


def test_tag_explanation_returns_one_claim_per_sentence():
    claims = tag_explanation("The text reports fighting. The next part may name the region.", INPUT)
    assert [c.kind for c in claims] == ["about", "expectation"]
