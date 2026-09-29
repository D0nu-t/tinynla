import json

from fastapi.testclient import TestClient

from nla.gui.dashboard import DashboardState, create_dashboard_app, parse_pipeline_log, parse_tqdm

LOG = (
    "==================================================\n"
    "[ar_sft] python -m training.train_ar_sft --config c.yaml\n"
    "==================================================\n"
    "Loading weights: 100%|##########| 148/148 [00:00<00:00, 2747.76it/s]\n"
    "epoch 0:  50%|#####     | 1/2 [00:01<00:01,  1.00s/it, loss=0.5]\r"
    "epoch 0: 100%|##########| 2/2 [00:02<00:00,  1.00s/it, loss=0.4]\n"
    "[ar_sft] epoch 0 val: {'fve': -0.1}\n"
    "Writing model shards: 100%|##########| 1/1 [00:01<00:00,  1.51s/it]\n"
    "==================================================\n"
    "[rl] python -m training.train_rl --config c.yaml\n"
    "==================================================\n"
    "grpo:   0%|          | 1/1500 [00:10<4:10:00, 10.00s/it, kl=0.001, ok=1.00, r=-0.55]\r"
    "grpo:   0%|          | 1/1500 [00:10<4:10:00, 10.00s/it, kl=0.001, ok=1.00, r=-0.55]\r"
    "grpo:   0%|          | 2/1500 [00:20<4:09:50, 10.00s/it, kl=0.002, ok=0.88, r=-0.66]\r"
)


def test_parse_tqdm_units_and_postfix():
    t = parse_tqdm("grpo:   1%|          | 15/1500 [02:34<4:14:52, 10.30s/it, kl=0.001, ok=1.00, r=-0.552]")
    assert (t["n"], t["total"], t["sec_per_it"]) == (15, 1500, 10.30)
    assert t["eta_s"] == 4 * 3600 + 14 * 60 + 52
    assert t["postfix"] == {"kl": "0.001", "ok": "1.00", "r": "-0.552"}
    fast = parse_tqdm("extract:  69%|######8   | 857/1250 [03:31<01:38,  4.00it/s]")
    assert abs(fast["sec_per_it"] - 0.25) < 1e-9


def test_pipeline_log_stages_progress_and_series():
    log = parse_pipeline_log(LOG)
    assert log["started"] == ["ar_sft", "rl"] and log["current"] == "rl"
    # housekeeping bars never replace the training bar
    assert log["progress"]["ar_sft"]["desc"] == "epoch 0"
    assert log["progress"]["rl"]["n"] == 2
    # repeated redraws of step 1 collapse to one point
    assert [r["n"] for r in log["series"]["rl"]] == [1, 2]
    assert log["series"]["rl"][1]["r"] == -0.66
    assert any("epoch 0 val" in m for m in log["messages"])


def test_failed_and_finished_flags():
    assert parse_pipeline_log(LOG + "[ERROR] stage rl failed\n")["failed"] == "rl"
    done = parse_pipeline_log(LOG + "[OK] Pipeline complete.\n")
    assert done["finished"] and done["current"] is None


def test_api_reads_files(tmp_path):
    log = tmp_path / "pipe.log"
    log.write_text(LOG, encoding="utf-8")
    cfg = {
        "experiment": {"name": "t_L1"},
        "model": {"target_name": "gpt2", "layer": 1},
        "data": {"output_dir": str(tmp_path / "data")},
        "ar_sft": {"save_dir": str(tmp_path / "ar")},
        "av_sft": {"save_dir": str(tmp_path / "av")},
        "rl": {"save_dir": str(tmp_path / "rl")},
    }
    (tmp_path / "rl").mkdir()
    (tmp_path / "rl" / "metrics.jsonl").write_text(
        json.dumps({"step": 0, "split": "val", "av_fve": -0.16}) + "\n{partial", encoding="utf-8")

    client = TestClient(create_dashboard_app(DashboardState(cfg, str(log))))
    assert "Training Dashboard" in client.get("/dashboard").text
    s = client.get("/api/dashboard/summary").json()
    assert s["stages"]["rl"] == "running" and s["stages"]["ar_sft"] == "done"
    m = client.get("/api/dashboard/metrics").json()
    assert m["jsonl"]["rl"] == [{"step": 0, "split": "val", "av_fve": -0.16}]   # partial line skipped
    assert len(m["live"]["rl"]) == 2
