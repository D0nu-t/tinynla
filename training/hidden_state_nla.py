"""
training/hidden_state_nla.py

Can the NLA reveal something that is in the model's state but NOT in the text?

We plant a concept inside the target with activation steering and leave the text
unchanged, so the ground truth is known and no text-based method can see it:
  - ActAdd (Turner et al. 2023): add a concept vector to the residual stream at
    an EARLIER layer (steer_layer < read layer); the vector here is a
    difference-in-means over short concept texts (as in CAA, Rimsky et al. 2024).
  - concept injection (Lindsey 2025, "Emergent introspective awareness"):
    same design, with the model itself as the reporter; here the NLA reports.

For each held-out context x concept x strength we record, at the read layer:
    nla       AV explanations of the steered activation   (the tool under test)
    lens      tuned/logit-lens top words at the read layer (independent method)
    behaviour the steered model's own greedy continuation  (did steering take?)
and identify the planted concept among all concepts by sentence-embedding
similarity (all-MiniLM-L6-v2), both absolutely and as a CHANGE from the same
context unsteered ("compared with normal, it is now thinking about X").
The text itself is identical under steering, so a text-only reader is at chance.

    python -m training.hidden_state_nla --config configs/gpt2_small.yaml
Outputs <av_dir>/hidden_state.json
"""

import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

import nla.model_adapter as ma
from nla.lens import top_words
from nla.read import NLAReader
from nla.utils import cli_config, set_seed
from study.simulate import minilm_embedder

load_dotenv()

# concept -> (texts that define the steering vector, description used for identification)
CONCEPTS = {
    "weddings": ("The bride and groom were married at a beautiful wedding. The wedding guests "
                 "celebrated the marriage ceremony.", "weddings, brides, grooms and marriage"),
    "football": ("The football players scored a goal in the stadium. The soccer match ended "
                 "with fans cheering for the team.", "football, soccer matches, goals and players"),
    "cooking": ("She baked bread in the kitchen and cooked dinner from a recipe. The food "
                "tasted delicious.", "cooking, recipes, kitchens and food"),
    "space": ("The astronauts launched a rocket into space to explore the planets. NASA studied "
              "the stars and galaxies.", "outer space, rockets, astronauts and planets"),
    "dogs": ("The puppy barked and wagged its tail. My dog loves walks and playing fetch "
             "with other dogs.", "dogs, puppies and pets"),
    "music": ("The band played guitar and drums at the concert. She sang her favourite song "
              "and the music was loud.", "music, songs, bands and concerts"),
    "medicine": ("The doctor examined the patient in the hospital. Nurses gave medicine to "
                 "treat the disease.", "hospitals, doctors, patients and medicine"),
    "computers": ("The programmer wrote software code on his computer. The program crashed "
                  "because of a bug in the software.", "computers, software and programming"),
    "ocean": ("Waves crashed on the beach as fish swam in the ocean. Sailors crossed the sea "
              "in a boat.", "the ocean, the sea, beaches and fish"),
    "weather": ("Heavy rain and strong winds hit the town as the storm arrived. Snow and "
                "thunder followed the storm.", "weather, storms, rain and snow"),
}


@contextmanager
def steer(model, layer: int, vector: torch.Tensor):
    """Add `vector` to the residual stream leaving block `layer` at every position."""
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            return (outputs[0] + vector,) + tuple(outputs[1:])
        return outputs + vector
    h = ma.blocks(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def concept_vectors(reader: NLAReader, layer: int) -> torch.Tensor:
    """Per-concept mean activation minus the mean over all concepts (unit length)."""
    means = []
    for text, _ in CONCEPTS.values():
        ids = reader.tokenizer(text, return_tensors="pt").input_ids.to(reader.device)
        with ma.capture_block_output(reader.target, layer) as cap:
            reader.target(ids)
        means.append(cap["hidden"][0, 1:].float().mean(0))     # skip position 0 (attention sink)
    m = torch.stack(means)
    return torch.nn.functional.normalize(m - m.mean(0, keepdim=True), dim=-1)


@torch.no_grad()
def typical_norm(reader: NLAReader, layer: int, texts) -> float:
    norms = []
    for t in texts:
        ids = reader.tokenizer(t, return_tensors="pt").input_ids.to(reader.device)
        with ma.capture_block_output(reader.target, layer) as cap:
            reader.target(ids)
        norms.append(cap["hidden"][0, 1:].float().norm(dim=-1))
    return torch.cat(norms).median().item()


def identify(sims: torch.Tensor, truth: int) -> dict:
    """sims [n_concepts]: top-1 hit and rank (1 = best) of the true concept."""
    rank = int((sims > sims[truth]).sum().item()) + 1
    return {"hit": rank == 1, "rank": rank}


def main():
    cfg, extra = cli_config(__doc__)
    set_seed(cfg["experiment"]["seed"], deterministic=False)
    n_ctx = int(next((a.split("=")[1] for a in extra if a.startswith("--n=")), 24))
    reader = NLAReader(cfg)
    read_layer = reader.layer
    steer_layer = max(1, read_layer // 2)
    strengths = [0.125, 0.25, 0.5, 1.0]  # added norm / median token norm at steer_layer
    control_strengths = [0.5, 1.0]       # random-direction control (false-alarm rate)
    n_samples = 3

    rng = random.Random(0)
    contexts = [reader.data.record["contexts"][i] for i in rng.sample(reader.data.split["test"], n_ctx)]
    contexts = [c[-400:] for c in contexts]            # keep prompts short and comparable

    vecs = concept_vectors(reader, steer_layer)
    g = torch.Generator().manual_seed(1)
    rand_vecs = torch.nn.functional.normalize(torch.randn(len(CONCEPTS), vecs.shape[1], generator=g), dim=-1)
    rand_vecs = rand_vecs.to(vecs.device)
    base_norm = typical_norm(reader, steer_layer, contexts[:8])
    names = list(CONCEPTS)
    embed = minilm_embedder(reader.device)
    concept_emb = embed([d for _, d in CONCEPTS.values()])
    print(f"[hidden] steer layer {steer_layer} -> read layer {read_layer}; median norm {base_norm:.1f}; "
          f"{n_ctx} contexts x {len(names)} concepts x {strengths}")

    tok = reader.tokenizer

    @torch.no_grad()
    def views(text, vector=None):
        ids = tok(text, return_tensors="pt").input_ids.to(reader.device)
        ctx = steer(reader.target, steer_layer, vector) if vector is not None else _null()
        with ctx:
            with ma.capture_block_output(reader.target, read_layer) as cap:
                reader.target(ids)
            h = cap["hidden"][0, -1].float()
            cont = reader.target.generate(ids, max_new_tokens=20, do_sample=False,
                                          pad_token_id=tok.eos_token_id)
        cont = tok.decode(cont[0, ids.shape[1]:], skip_special_tokens=True)
        expl = [o["text"] for o in reader.av.generate(h.unsqueeze(0).to(reader.device), n_samples=n_samples)]
        lens_words = [w["word"] for w in top_words(reader.lens(h.to(reader.device)).float().cpu(), tok, k=20)]
        return {"nla": " ".join(expl), "lens": ", ".join(lens_words), "behaviour": cont}

    methods = ["nla", "lens", "behaviour"]
    rows, examples = [], []
    for ci, text in enumerate(tqdm(contexts, desc="contexts")):
        base = views(text)
        base_e = {m: embed([base[m]])[0] for m in methods}
        conds = [(k, s, False) for k in range(len(names)) for s in strengths] +                 [(k, s, True) for k in range(len(names)) for s in control_strengths]
        for k, s, random_dir in conds:
                name = names[k]
                vec = rand_vecs[k] if random_dir else vecs[k]
                v = views(text, (s * base_norm * vec).to(reader.device, reader.target.dtype))
                # random rows keep label k only to measure how often a method "finds" it anyway
                row = {"context": ci, "concept": name, "strength": s, "random": random_dir}
                for m in methods:
                    e = embed([v[m]])[0]
                    abs_sims = concept_emb @ e
                    delta_sims = concept_emb @ (e - base_e[m])
                    row[m] = identify(abs_sims, k)
                    row[m + "_change"] = identify(delta_sims, k)
                rows.append(row)
                if s == 0.5 and ci < 3 and not random_dir:
                    examples.append({"context_tail": text[-120:], "concept": name,
                                     "nla": v["nla"][:300], "lens": v["lens"],
                                     "behaviour": v["behaviour"], "unsteered_nla": base["nla"][:300]})

    chance = 1 / len(names)
    summary = {"chance": chance, "steer_layer": steer_layer, "read_layer": read_layer,
               "n_contexts": n_ctx, "concepts": names, "by_strength": {}}
    print(f"\nPlanted concept identified (top-1 of {len(names)}; chance {chance:.2f}; ± s.e.)")
    for s, random_dir in [(s, False) for s in strengths] + [(s, True) for s in control_strengths]:
        sub = [r for r in rows if r["strength"] == s and r["random"] == random_dir]
        res = {}
        for m in methods:
            for kind in ("", "_change"):
                hits = [r[m + kind]["hit"] for r in sub]
                p = sum(hits) / len(hits)
                res[m + kind] = {"rate": p, "se": math.sqrt(p * (1 - p) / len(hits))}
        # does the NLA see it when the behaviour shows it (and when it doesn't)?
        took = [r for r in sub if r["behaviour_change"]["hit"]]
        hidden = [r for r in sub if not r["behaviour_change"]["hit"]]
        res["nla_change_when_behaviour_shows"] = (sum(r["nla_change"]["hit"] for r in took) / len(took)) if took else None
        res["nla_change_when_behaviour_hides"] = (sum(r["nla_change"]["hit"] for r in hidden) / len(hidden)) if hidden else None
        res["n"] = len(sub)
        key = f"random_{s}" if random_dir else str(s)
        summary["by_strength"][key] = res
        line = "  ".join(f"{m}{kind} {res[m + kind]['rate']:.2f}" for m in methods for kind in ("", "_change"))
        print(f"  {key:>10}: {line}")
        print(f"      NLA (change) when behaviour shows it: {res['nla_change_when_behaviour_shows']}"
              f" | when behaviour doesn't: {res['nla_change_when_behaviour_hides']}")
    summary["examples"] = examples
    out = Path(reader.av_dir) / "hidden_state.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for ex in examples[:6]:
        print(f"\n  [{ex['concept']}] ...{ex['context_tail'][-80:]!r}")
        print(f"    behaviour: {ex['behaviour'][:100]!r}")
        print(f"    NLA:       {ex['nla'][:160]!r}")
        print(f"    lens:      {ex['lens'][:100]}")
    print(f"[OK] {out}")


@contextmanager
def _null():
    yield


if __name__ == "__main__":
    main()
