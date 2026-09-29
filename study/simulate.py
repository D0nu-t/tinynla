"""
study/simulate.py

Meaning-based simulated participants for the pilot (an information check run
BEFORE recruiting people; not a substitute for them).

A word-overlap matcher is too blunt for gist-level reports: on most items the
report shares no word with either option. This one compares sentence embeddings
(all-MiniLM-L6-v2) and scores each reader by its *margin*: cos(query, real
option) - cos(query, distractor). The key numbers subtract a control where each
item gets ANOTHER item's report (averaged over shuffles), so any generic "sounds
like a report" advantage cancels:

    report_info         report alone, real - shuffled margin   (does it carry item info?)
    beyond_prompt       prompt+report, real - shuffled margin  (more than a random report adds?)
    beyond_prompt_ambiguous   same, on the half where the prompt alone is least decisive

    python -m study.simulate study/pilot/items.json
"""

import json
import math
import sys
from typing import Callable, Dict, List

import torch

from nla.embed import EMBED_MODEL, minilm_embedder  # noqa: F401  (re-exported)


def report_text(report: Dict) -> str:
    """What a reader actually sees: the narrative plus the theme words."""
    words = ", ".join(t.get("word", t["stem"]) for t in report["themes"])
    return f"{report['summary']} Themes: {words}"


def _mean_se(x: torch.Tensor) -> Dict[str, float]:
    return {"mean": x.mean().item(), "se": (x.std() / math.sqrt(len(x))).item() if len(x) > 1 else float("nan")}


def embedding_check(items: List[Dict], embed: Callable[[List[str]], torch.Tensor],
                    n_shuffles: int = 20, seed: int = 0) -> Dict:
    n = len(items)
    ans = torch.tensor([int(it["answer"]) for it in items])
    O = embed([o for it in items for o in it["options"]]).view(n, 2, -1)
    P = embed([it["prompt"] for it in items])
    R = embed([report_text(it["report"]) for it in items])

    def margin(q):
        s = torch.einsum("nd,nkd->nk", q, O)
        return (s.gather(1, ans[:, None]) - s.gather(1, 1 - ans[:, None])).squeeze(1)

    def with_prompt(q):
        return torch.nn.functional.normalize(P + q, dim=-1)

    g = torch.Generator().manual_seed(seed)
    # derangement-free shuffles are fine: a self-match only dilutes the control
    shuffles = [R[torch.randperm(n, generator=g)] for _ in range(n_shuffles)]
    m_p, m_r, m_pr = margin(P), margin(R), margin(with_prompt(R))
    m_rs = torch.stack([margin(s) for s in shuffles]).mean(0)
    m_prs = torch.stack([margin(with_prompt(s)) for s in shuffles]).mean(0)
    ambiguous = m_p.abs() < m_p.abs().median()

    return {
        "n": n,
        "accuracy": {"prompt": (m_p > 0).float().mean().item(),
                     "report": (m_r > 0).float().mean().item(),
                     "prompt+report": (m_pr > 0).float().mean().item()},
        "report_info": _mean_se(m_r - m_rs),
        "beyond_prompt": _mean_se(m_pr - m_prs),
        "beyond_prompt_ambiguous": _mean_se((m_pr - m_prs)[ambiguous]),
    }


def describe(res: Dict) -> str:
    def z(k):
        return res[k]["mean"] / res[k]["se"] if res[k]["se"] else float("nan")
    a = res["accuracy"]
    return "\n".join([
        f"n={res['n']}  accuracy: prompt {a['prompt']:.3f} | report {a['report']:.3f} | "
        f"prompt+report {a['prompt+report']:.3f}",
        f"  report carries item info (real - shuffled):   {res['report_info']['mean']:+.4f} "
        f"± {res['report_info']['se']:.4f}  (z={z('report_info'):.1f})",
        f"  prompt+report vs prompt+shuffled report:      {res['beyond_prompt']['mean']:+.4f} "
        f"± {res['beyond_prompt']['se']:.4f}  (z={z('beyond_prompt'):.1f})",
        f"    ...where the prompt alone is ambiguous:     {res['beyond_prompt_ambiguous']['mean']:+.4f} "
        f"± {res['beyond_prompt_ambiguous']['se']:.4f}  (z={z('beyond_prompt_ambiguous'):.1f})",
    ])


def main():
    from nla.utils import utf8_stdio
    utf8_stdio()
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    embed = minilm_embedder()
    for path in sys.argv[1:]:
        items = json.load(open(path, encoding="utf-8"))["items"]
        print(path)
        print(describe(embedding_check(items, embed)))


if __name__ == "__main__":
    main()
