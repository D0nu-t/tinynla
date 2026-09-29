# Pilot study kit: does the report help people predict the model?

A forward-simulation study (Hase & Bansal 2020; human-grounded evaluation, Doshi-Velez & Kim 2017), with an over-trust check (Bansal et al. 2021).

1. **Build items** (needs the trained NLA): `python -m study.make_pilot --n 32 --out study/pilot`. It writes:
   - `items.json`: items plus answers. Keep it private.
   - `pilot_form_A.html` / `pilot_form_B.html`: self-contained questionnaires. Each shows the report on half of the items, and the two forms are counterbalanced.
2. **Collect.** Send each participant one form (alternate A and B). They copy the JSON shown at the end back to you. No personal data is collected.
3. **Score** with `python -m study.score_pilot study/pilot/items.json responses/*.json`, which reports:
   - accuracy with vs without the report;
   - the paired gain with a sign test across participants;
   - over-confidence (how much surer people felt than they were).

**Distractors.** `--distractor unlikely_sample` (the default) uses a coherent nucleus sample from the same model that diverges from its most likely continuation. `other_model` uses distilgpt2, which is same-topic and too hard for a gist-level report.

**Check before recruiting.** `make_pilot` runs the meaning-based check in `study/simulate.py` (re-run it on any `items.json` with `python -m study.simulate study/pilot/items.json`). It compares sentence embeddings, and its control gives each item another item's report. Trust it over the word-overlap matchers, which tie on most items.

At GPT-2 scale (200 items, continued-RL NLA):
- reports carry clear item-specific information (6σ above the shuffled control);
- they add **nothing detectable beyond the prompt**.

So expect little accuracy gain from people. Use this kit mainly to measure over-trust, or to evaluate a stronger NLA.
