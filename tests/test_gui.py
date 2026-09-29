import json

from fastapi.testclient import TestClient

from nla.gui.server import create_app


class FakeReader:
    def run_info(self):
        return {"target": "gpt2", "layer": 8, "stage": "sft", "av_dir": "x", "ar_dir": "y"}

    def tokenize(self, text):
        return [
            {"position": i, "token": w, "norm": 100.0 + i, "eligible": i >= 2, "top_next": "."}
            for i, w in enumerate(text.split())
        ]

    def read(self, text, position, n_samples):
        for k in range(n_samples):
            yield {"index": k, "position": position, "text": f"about {text[:5]}",
                   "ok": True, "fve": 0.4, "cosine": 0.9}

    def controls(self, text, position):
        return {"position": position, "mean_fve": 0.0, "shuffled_fve": -0.3, "norm": 120.0}

    def report(self, prompt, answer=None, n_samples=3):
        if answer == " ":
            raise ValueError("answer has no readable tokens")
        return {"prompt": prompt, "answer": answer or " model answer", "overall_strength": "moderate",
                "summary": "…", "themes": [], "tokens": [], "limits": ["x"]}

    def contrast(self, prompt, baseline, answer=None, n_samples=4):
        if prompt == baseline:
            raise ValueError("identical prompts")
        return {"prompt": prompt, "baseline": baseline, "answer": answer or " model answer",
                "summary": "…", "more": [], "less": [], "lens_more": [], "limits": ["x"],
                "tokens_read": 4, "n_target_reads": 12, "n_baseline_reads": 12}

    def compare_vectors(self, text, position, explanation):
        return {"dims": [1, 2], "gold": [0.5, -0.2], "recon": [0.4, 0.1], "fve": 0.4, "cosine": 0.9}


client = TestClient(create_app(FakeReader()))
TEXT = "one two three four five"


def test_index_and_static():
    assert "Thought Reader" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_run_and_tokenize():
    assert client.get("/api/run").json()["layer"] == 8
    toks = client.post("/api/tokenize", json={"text": TEXT}).json()["tokens"]
    assert len(toks) == 5 and toks[0]["eligible"] is False


def test_read_streams_one_event_per_sample_then_done():
    res = client.post("/api/read", json={"text": TEXT, "position": 3, "n_samples": 3})
    assert res.headers["content-type"].startswith("text/event-stream")
    events = [e for e in res.text.split("\n\n") if e.strip()]
    data = [json.loads(e[len("data: "):]) for e in events if e.startswith("data: ")]
    assert [d["index"] for d in data] == [0, 1, 2]
    assert events[-1].startswith("event: done")


def test_position_out_of_range_is_422():
    res = client.post("/api/read", json={"text": TEXT, "position": 9, "n_samples": 1})
    assert res.status_code == 422


def test_controls_and_compare():
    c = client.post("/api/controls", json={"text": TEXT, "position": 3}).json()
    assert c["shuffled_fve"] < c["mean_fve"]
    v = client.post("/api/compare", json={"text": TEXT, "position": 3, "explanation": "x"}).json()
    assert len(v["dims"]) == len(v["gold"]) == len(v["recon"])


def test_report_page_and_endpoint():
    assert "What was the model thinking?" in client.get("/report").text
    r = client.post("/api/report", json={"prompt": "Once upon a time"}).json()
    assert r["answer"] == " model answer" and r["overall_strength"] == "moderate"
    assert client.post("/api/report", json={"prompt": "x", "answer": " "}).status_code == 422
    assert client.post("/api/report", json={"prompt": ""}).status_code == 422


def test_contrast_page_and_api():
    assert "different from normal" in client.get("/compare").text
    assert client.get("/static/compare.js").status_code == 200
    r = client.post("/api/contrast", json={"prompt": "a hint. text", "baseline": "text"})
    assert r.status_code == 200 and r.json()["answer"] == " model answer"
    assert client.post("/api/contrast", json={"prompt": "x", "baseline": "x"}).status_code == 422
    assert client.post("/api/contrast", json={"prompt": "x"}).status_code == 422
