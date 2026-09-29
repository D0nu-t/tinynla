"""
nla/read.py

The utility: read a model's "thoughts" at chosen tokens.

    python -m nla.read "The capital of France is" --position -1 --samples 3

    reader = NLAReader.from_config("configs/gpt2_small.yaml")
    for token in reader.tokenize(text): ...
    for expl in reader.read(text, position=12, n_samples=3): ...

Every explanation carries an AR self-check: the explanation is fed back
through the AR and compared with the true activation. `fve` is per-sample
FVE, 1 - ||g - p||^2 / Var_train, on the same scale as the dataset metric:
0 = no better than the average activation, 1 = perfect.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from nla import model_adapter as ma
from nla.ar import ActivationReconstructor
from nla.av import ActivationVerbalizer
from nla.calibrate import Calibration
from nla.contrast import LIMITS as CONTRAST_LIMITS, contrast_summary, contrast_themes
from nla.lens import Lens, difference_lens, logit_lens, top_words
from nla.metrics import per_sample_fve, rescale
from nla.report import build_report
from nla.unusual import UnusualnessModel
from nla.runs import DataBundle, load_data, load_target, model_dirs
from nla.utils import load_config, resolve_device, utf8_stdio


class NLAReader:
    def __init__(self, cfg: Dict, stage: str = "auto", device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or resolve_device(cfg)
        self.layer = cfg["model"]["layer"]
        self.min_position = cfg["data"]["min_position"]

        name = cfg["model"]["target_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.target = load_target(cfg, self.device)

        self.av_dir, self.ar_dir = model_dirs(cfg, stage)
        self.av = ActivationVerbalizer.load(self.av_dir).to(self.device).eval()
        self.ar = ActivationReconstructor.load(self.ar_dir).to(self.device).eval()

        self.data: DataBundle = load_data(cfg)
        # RL checkpoints are <rl.save_dir>/av; SFT ones are av_sft.save_dir
        self.stage = "rl" if Path(self.av_dir).name == "av" else "sft"
        self.calibration = Calibration.load(self.av_dir)   # None until calibrate_nla runs
        self.lens = Lens(self.target, cfg["model"]["target_name"], self.layer)
        # "unusual state" detector (nla/unusual.py), fitted once per dataset and cached
        self.unusual = UnusualnessModel.for_data(self.data, Path(cfg["data"]["output_dir"]) / "unusual.pt")

    @classmethod
    def from_config(cls, path: str, stage: str = "auto", **kw) -> "NLAReader":
        return cls(load_config(path), stage=stage, **kw)

    # ------------------------------------------------------------------
    # Target model
    # ------------------------------------------------------------------

    @torch.no_grad()
    def activations(self, text: str) -> Dict:
        ids = torch.tensor([self._ids(text)], device=self.device)
        with ma.capture_block_output(self.target, self.layer) as cap:
            out = self.target(ids)
        hidden = cap["hidden"][0].float().cpu()
        next_ids = out.logits[0].argmax(dim=-1).tolist()
        return {"ids": ids[0].tolist(), "hidden": hidden, "next": next_ids}

    def tokenize(self, text: str) -> List[Dict]:
        a = self.activations(text)
        norms = a["hidden"].norm(dim=-1).tolist()
        return [
            {
                "position": i,
                "token": self.tokenizer.decode([t]),
                "norm": norms[i],
                "eligible": i >= self.min_position,
                "top_next": self.tokenizer.decode([a["next"][i]]),
            }
            for i, t in enumerate(a["ids"])
        ]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, explanations: List[str], gold: torch.Tensor) -> List[Dict]:
        """Per-explanation self-check against one activation [d]."""
        with torch.no_grad(), torch.autocast(self.ar.device.type,
                                             enabled=self.ar.device.type == "cuda"):
            pred = self.ar(explanations).float().cpu()
        g = rescale(gold.unsqueeze(0), self.data.scale)
        p = rescale(pred, self.data.scale)
        err = ((p - g) ** 2).sum(dim=-1)
        return [
            {
                "fve": float(1 - e / self.data.variance),
                "cosine": float(F.cosine_similarity(p[i], g[0], dim=0)),
                "reconstruction": pred[i],
            }
            for i, e in enumerate(err)
        ]

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self, text: str, position: int = -1, n_samples: int = 3) -> Iterator[Dict]:
        """Yield explanations one at a time (for streaming)."""
        a = self.activations(text)
        pos = position % len(a["ids"])
        gold = a["hidden"][pos]
        for k in range(n_samples):
            out = self.av.generate(gold.unsqueeze(0), n_samples=1)[0]
            s = self.score([out["text"]], gold)[0]
            yield {
                "index": k,
                "position": pos,
                "text": out["text"],
                "ok": out["ok"],
                "fve": s["fve"],
                "cosine": s["cosine"],
            }

    def controls(self, text: str, position: int = -1, n_shuffled: int = 8) -> Dict:
        """
        Touchstone controls for one token:
            mean       predicting the train mean                -> 0 by construction
            shuffled   random training explanations, averaged    -> should be <= 0
            top_dims   largest |true - reconstruction| dims from one AV sample
        """
        a = self.activations(text)
        pos = position % len(a["ids"])
        gold = a["hidden"][pos]

        rng = random.Random(pos)
        pool = rng.sample(self.data.split["train"], n_shuffled)
        shuffled = self.score([self.data.explanations[i] for i in pool], gold)

        g = rescale(gold.unsqueeze(0), self.data.scale)[0]
        mean_err = float(((g - self.data.mean) ** 2).sum())
        return {
            "position": pos,
            "mean_fve": 1 - mean_err / self.data.variance,
            "shuffled_fve": sum(s["fve"] for s in shuffled) / len(shuffled),
            "norm": float(gold.norm()),
        }

    # ------------------------------------------------------------------
    # Answer-level report (stakeholder view)
    # ------------------------------------------------------------------

    def read_many(self, text: str, positions: List[int], n_samples: int = 3,
                  batch: int = 8) -> List[Dict]:
        """Batched reads at several positions: [{position, token, explanations:[{text, ok, fve}]}]."""
        a = self.activations(text)
        vectors = a["hidden"][positions]
        v = vectors.to(self.device)
        # two independent views, two jobs (measured in training/anticipation_nla.py):
        #   lens       - tuned lens: best forecast of what the model will say (display)
        #   lens_check - logit lens: agreement with it best predicts a correct NLA read (corroboration)
        forecast, check = self.lens(v), logit_lens(self.target, v)
        readings = [{"position": p, "token": self.tokenizer.decode([a["ids"][p]]), "explanations": [],
                     "lens": top_words(forecast[j], self.tokenizer, k=10),
                     "lens_check": top_words(check[j], self.tokenizer, k=10),
                     "unusual": u}
                    for j, (p, u) in enumerate(zip(positions, self.unusual.assess(vectors)))]
        for start in range(0, len(positions), batch):
            chunk = vectors[start:start + batch]
            outs = self.av.generate(chunk, n_samples=n_samples)
            with torch.no_grad(), torch.autocast(self.ar.device.type,
                                                 enabled=self.ar.device.type == "cuda"):
                pred = self.ar([o["text"] for o in outs]).float().cpu()
            golds = chunk.repeat_interleave(n_samples, dim=0)
            fves = per_sample_fve(pred, golds, self.data.variance, self.data.scale).tolist()
            for o, f in zip(outs, fves):
                readings[start + o["source"]]["explanations"].append(
                    {"text": o["text"], "ok": o["ok"], "fve": f})
        return readings

    @torch.no_grad()
    def _ids(self, text: str) -> List[int]:
        # a chat template already contains any BOS token; adding another would
        # shift every position (Llama-style templates)
        return self.tokenizer(text, add_special_tokens=not self.chat).input_ids

    @property
    def chat(self) -> bool:
        """Chat models answer inside their chat template (config model.chat, default on)."""
        return bool(getattr(self.tokenizer, "chat_template", None)) and self.cfg["model"].get("chat", True)

    def model_view(self, prompt: str) -> str:
        """The text the model actually sees before it answers `prompt`."""
        if not self.chat:
            return prompt
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                  tokenize=False, add_generation_prompt=True)

    def complete(self, prompt: str, max_new_tokens: int = 24) -> str:
        """Let the target model write its own answer (greedy; in chat format for chat models)."""
        toks = self.tokenizer(self.model_view(prompt), return_tensors="pt",
                              add_special_tokens=not self.chat).to(self.device)
        out = self.target.generate(**toks, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, toks["input_ids"].shape[1]:], skip_special_tokens=True)

    def report(self, prompt: str, answer: Optional[str] = None, n_samples: int = 3,
               max_tokens: int = 24) -> Dict:
        """
        "What was the model thinking while it wrote this answer?"

        Reads the answer's tokens (evenly thinned to `max_tokens`), then
        aggregates themes, claim labels and calibrated strength (nla.report).
        """
        if answer is None:
            answer = self.complete(prompt)
        seen = self.model_view(prompt)            # positions refer to what the model saw
        text = seen + answer
        n_prompt = len(self._ids(seen))
        n_total = len(self._ids(text))
        positions = [p for p in range(max(n_prompt, self.min_position), n_total)]
        if len(positions) > max_tokens:
            step = len(positions) / max_tokens
            positions = [positions[int(i * step)] for i in range(max_tokens)]
        if not positions:
            raise ValueError(f"answer has no readable tokens (positions < {self.min_position} are skipped)")

        readings = self.read_many(text, positions, n_samples)
        ids = self._ids(text)
        # claims are judged against the user's words, not the chat template's boilerplate
        rep = build_report(readings, prompt + answer, self.calibration,
                           token_texts=[self.tokenizer.decode([t]) for t in ids],
                           flag_quantile=self.unusual.flag_quantile)
        rep.update({"prompt": prompt, "answer": answer, "lens_kind": self.lens.kind, "chat_format": self.chat,
                    "model": self.cfg["model"]["target_name"],
                    "layer": self.layer, "stage": self.stage})
        return rep

    def contrast(self, prompt: str, baseline: str, answer: Optional[str] = None, n_samples: int = 4,
                 max_tokens: int = 16) -> Dict:
        """
        "What is different from normal?" Read the SAME answer tokens after `prompt`
        and after `baseline` (e.g. the prompt without a suspected hint), and report
        the themes and lens words that differ (nla.contrast). The answer defaults to
        the model's own continuation of `prompt`, teacher-forced under both.
        """
        if answer is None:
            answer = self.complete(prompt)
        spans = []
        for p in (prompt, baseline):
            p = self.model_view(p)
            text = p + answer
            n_p, n_t = len(self._ids(p)), len(self._ids(text))
            spans.append((text, list(range(max(n_p, self.min_position), n_t))))
        k = min(len(spans[0][1]), len(spans[1][1]))
        if k == 0:
            raise ValueError("answer has no readable tokens under one of the prompts")
        picks = sorted({int(i * k / min(k, max_tokens)) for i in range(min(k, max_tokens))})
        # align answer tokens from the END (prompt-boundary tokenisation may differ by one)
        pos = [[s[len(s) - k + i] for i in picks] for _, s in spans]
        reads = [self.read_many(text, p, n_samples) for (text, _), p in zip(spans, pos)]

        if not hasattr(self, "_embed"):
            from nla.embed import minilm_embedder
            self._embed = minilm_embedder(self.device)
        res = contrast_themes(reads[0], reads[1], self.calibration, prompt, baseline, embed=self._embed)
        h_t = self.activations(spans[0][0])["hidden"][pos[0]]
        h_b = self.activations(spans[1][0])["hidden"][pos[1]]
        d = (h_t - h_b).mean(0).to(self.device)
        res["lens_more"] = top_words(difference_lens(self.target, d), self.tokenizer, k=8, whole_words=True)
        res["lens_less"] = top_words(difference_lens(self.target, -d), self.tokenizer, k=8, whole_words=True)
        res["summary"] = contrast_summary(res)
        res.update({"prompt": prompt, "baseline": baseline, "answer": answer,
                    "tokens_read": len(picks), "limits": CONTRAST_LIMITS,
                    "model": self.cfg["model"]["target_name"], "layer": self.layer})
        return res

    def compare_vectors(self, text: str, position: int, explanation: str, k: int = 16) -> Dict:
        """Gold vs reconstruction on the dims where they differ most (for the GUI)."""
        a = self.activations(text)
        gold = a["hidden"][position % len(a["ids"])]
        s = self.score([explanation], gold)[0]
        g = rescale(gold.unsqueeze(0), self.data.scale)[0]
        p = rescale(s["reconstruction"].unsqueeze(0), self.data.scale)[0]
        dims = (g - p).abs().topk(k).indices.tolist()
        return {
            "dims": dims,
            "gold": [float(g[d]) for d in dims],
            "recon": [float(p[d]) for d in dims],
            "fve": s["fve"],
            "cosine": s["cosine"],
        }

    def run_info(self) -> Dict:
        info = {
            "target": self.cfg["model"]["target_name"],
            "layer": self.layer,
            "stage": self.stage,
            "av_dir": self.av_dir,
            "ar_dir": self.ar_dir,
            "injection_scale": self.av.injection_scale,
            "device": self.device,
            "calibrated": self.calibration is not None,
        }
        report = Path(self.av_dir) / "nla_eval.json"
        if report.exists():
            r = json.loads(report.read_text())
            info["eval"] = {k: r[k] for k in ("av_fve", "av_fve_shuffled", "summary_fve", "n_test")}
        return info


def main():
    utf8_stdio()
    p = argparse.ArgumentParser(description="Read a model's thoughts at a token.")
    p.add_argument("text")
    p.add_argument("--config", default="configs/gpt2_small.yaml")
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--stage", default="auto", choices=["auto", "sft", "rl"])
    p.add_argument("--report", action="store_true",
                   help="stakeholder report: treat TEXT as a prompt, let the model answer (or pass --answer)")
    p.add_argument("--answer", help="answer text to explain (default: the model writes one)")
    args = p.parse_args()

    reader = NLAReader.from_config(args.config, stage=args.stage)

    if args.report:
        rep = reader.report(args.text, args.answer, n_samples=args.samples)
        print(f"PROMPT : {rep['prompt']}\nANSWER : {rep['answer']}\n")
        print(f"SIGNAL : {rep['overall_strength_text']}")
        print(f"         {rep['coverage_text']}")
        print(f"         {rep['consistency']['text']}\n")
        print(rep["summary"] + "\n")
        for t in rep["themes"]:
            where = "in text" if t["in_input"] else "NOT in text"
            early = f", seen {t['words_ahead']} words before it was written" if t.get("anticipated") else ""
            early += ", confirmed by the lens" if t.get("corroborated") else ""
            print(f"  - {t['theme']:<16} {t['share_of_explanations']:>4.0%} of reads, {where}{early}")
        print("\nLimits:")
        for line in rep["limits"]:
            print(f"  * {line}")
        return

    toks = reader.tokenize(args.text)
    pos = args.position % len(toks)
    print(f"Reading token {pos}: {toks[pos]['token']!r}  (layer {reader.layer}, {reader.stage})")
    if not toks[pos]["eligible"]:
        print(f"  warning: positions < {reader.min_position} decode to the prior")
    for e in reader.read(args.text, pos, args.samples):
        print(f"  [{e['fve']:+.3f} FVE] {e['text']}")
    c = reader.controls(args.text, pos)
    print(f"  controls: mean {c['mean_fve']:+.3f}, shuffled {c['shuffled_fve']:+.3f}")


if __name__ == "__main__":
    main()
