"""
scripts/capture_gui.py

Screenshot the Thought Reader and the training dashboard for documentation.

Start the servers first:
    python -m nla.gui --no-browser --port 8000
    python -m nla.gui.dashboard --no-browser --port 8001
Then:
    python scripts/capture_gui.py --out <dir>

Uses Playwright with the installed Microsoft Edge (no browser download).
Also writes readings.json with the explanations shown in each screenshot.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

READER = "http://127.0.0.1:8000"
DASHBOARD = "http://127.0.0.1:8001/dashboard"

EXAMPLES = [
    ("story", "The old lighthouse keeper climbed the stairs every night to light the lamp, "
              "because he knew that somewhere out on the dark water a ship was"),
    ("news", "After three days of negotiations, the two parties finally agreed to a ceasefire, "
             "but within hours reports emerged that fighting had resumed in the"),
]
STRIP = ("recipe", "Preheat the oven to 180 degrees. In a large bowl, whisk together the flour, "
                   "sugar and baking powder, then slowly add the warm milk and")


def wait_for(url: str, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except Exception:
            time.sleep(3)
    raise TimeoutError(url)


def tokenize(page, text: str) -> None:
    page.goto(READER)
    page.wait_for_selector(".tok")
    page.fill("#text", text)
    page.click("#analyze")
    page.wait_for_function("document.querySelectorAll('.tok').length > 0 && !document.getElementById('analyze').disabled")
    page.wait_for_timeout(300)


def read_selected(page, n: int) -> list:
    page.select_option("#samples", str(n))
    page.click("#read")
    page.wait_for_function(
        f"document.querySelectorAll('.expl:not(.pending)').length >= {n} "
        "&& !document.getElementById('read').disabled",
        timeout=180_000,
    )
    page.wait_for_selector("#controls svg")
    return page.eval_on_selector_all(".expl:not(.pending)", "els => els.map(e => e.innerText)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scheme", default="dark", choices=["dark", "light"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wait_for(READER + "/api/run")
    wait_for(DASHBOARD)
    readings = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        ctx = browser.new_context(viewport={"width": 1280, "height": 1000},
                                  color_scheme=args.scheme, device_scale_factor=1.25)
        page = ctx.new_page()

        # 1. full reads: last token, 3 samples, then the AV/AR comparison
        for name, text in EXAMPLES:
            tokenize(page, text)
            texts = read_selected(page, 3)
            page.click(".expl:not(.pending)")
            page.wait_for_selector("#compare svg", timeout=60_000)
            page.wait_for_timeout(500)
            path = out / f"tinynla_gui_reader_{name}.png"
            page.screenshot(path=str(path), full_page=True)
            readings[name] = {"text": text, "explanations": texts}
            print("saved", path)

        # 2. token strip coloured by self-check FVE after reading several tokens
        name, text = STRIP
        tokenize(page, text)
        eligible = page.eval_on_selector_all(".tok:not(.ineligible)", "els => els.length")
        strip = []
        for k in range(min(6, eligible)):
            if k:
                # arrow keys are ignored while a <select> has focus
                page.locator(".panel-head h2").click()
                page.keyboard.press("ArrowLeft")
            texts = read_selected(page, 1)
            sel = page.inner_text("#selected")
            strip.append({"selected": sel, "explanation": texts[0]})
        page.select_option("#colorBy", "fve")
        page.wait_for_timeout(500)
        path = out / f"tinynla_gui_reader_{name}_fve_strip.png"
        page.screenshot(path=str(path), full_page=True)
        readings[name] = {"text": text, "tokens": strip}
        print("saved", path)

        # 3. plain-language report (stakeholder view), model writes its own answer
        page.goto(READER + "/report")
        page.fill("#prompt", EXAMPLES[1][1].rsplit(" in the", 1)[0])
        page.click("#go")
        page.wait_for_selector("#result:not(.hidden)", timeout=300_000)
        page.wait_for_timeout(500)
        path = out / "tinynla_gui_report.png"
        page.screenshot(path=str(path), full_page=True)
        readings["report"] = {"verdict": page.inner_text("#verdict"), "summary": page.inner_text("#summary"),
                              "coverage": page.inner_text("#coverage"), "answer": page.input_value("#answer")}
        print("saved", path)

        # 3. training dashboard for the finished run
        page.goto(DASHBOARD)
        page.wait_for_selector("#rlval svg", timeout=60_000)
        page.wait_for_selector("#final .checks")
        page.wait_for_timeout(1500)
        path = out / "tinynla_gui_dashboard.png"
        page.screenshot(path=str(path), full_page=True)
        print("saved", path)

        browser.close()

    (out / "tinynla_gui_readings.json").write_text(json.dumps(readings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
