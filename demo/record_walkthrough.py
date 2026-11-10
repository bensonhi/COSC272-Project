"""Record a silent backup video of the demo with headless Chromium (Playwright).

    PLAYWRIGHT_BROWSERS_PATH=/scratch/f008k3y/cosc272/pw python demo/record_walkthrough.py \
        --url http://127.0.0.1:7860 --out demo/recording

Walkthrough: each curated example (representative, challenging, failure) is verified, then its
predicted evidence is removed and re-verified, and all models are compared; finally batch mode.
Produces a .webm in --out (convert with ffmpeg if an .mp4 is required).
"""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

EXAMPLES = Path(__file__).with_name("examples.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--out", default=str(Path(__file__).with_name("recording")))
    parser.add_argument("--pause", type=int, default=3500, help="ms to hold each result on screen")
    args = parser.parse_args()
    examples = json.loads(EXAMPLES.read_text())
    size = {"width": 1440, "height": 900}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport=size, record_video_dir=args.out, record_video_size=size)
        page = context.new_page()
        page.goto(args.url)
        page.wait_for_selector("text=Curated examples")
        for example in examples:
            box = page.get_by_label("Curated examples (representative / challenging / failure)")
            box.click()
            box.fill(f"#{example['claim_id']}:")
            page.wait_for_timeout(600)
            page.keyboard.press("ArrowDown")
            page.keyboard.press("Enter")
            page.wait_for_timeout(1200)
            page.get_by_role("button", name="Verify", exact=True).click()
            page.wait_for_selector("text=tokens read", timeout=60000)
            page.mouse.wheel(0, 250)
            page.wait_for_timeout(args.pause)
            page.get_by_role("button", name="Compare all models").click()
            page.wait_for_timeout(args.pause)
            page.get_by_role("button", name="Remove predicted evidence & re-verify").click()
            page.wait_for_timeout(args.pause)
            page.mouse.wheel(0, -2000)
        page.get_by_role("tab", name="Batch mode").click()
        page.get_by_role("button", name="Run batch").click()
        page.wait_for_selector("text=median", timeout=600000)
        page.wait_for_timeout(args.pause * 2)
        page.get_by_role("tab", name="About").click()
        page.wait_for_timeout(args.pause * 2)
        context.close()
        browser.close()
    print("saved video(s) to", args.out)


if __name__ == "__main__":
    main()
