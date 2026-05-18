"""
main.py — Entrypoint for the weekly market report pipeline.
Usage:
    python scripts/main.py
"""
import os, sys, json, logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root, "output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Fetch data ──────────────────────────────────────────
    log.info("═══ Step 1/2: Fetching market data ═══")
    from fetch_data import run as fetch_run
    data = fetch_run()

    data_path = os.path.join(output_dir, "data.json")
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Data saved → {data_path}")

    # ── Step 2: Generate HTML report ────────────────────────────────
    log.info("═══ Step 2/2: Generating HTML report ═══")
    from generate_report import generate
    html = generate(data)

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"Report saved → {html_path}")
    log.info("✅ Done.")

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
