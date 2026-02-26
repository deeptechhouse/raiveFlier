# ─── raiveFeeder module entry point ────────────────────────────────────
# Enables: python -m tools.raive_feeder
# This delegates to main.py's main() function which creates the FastAPI
# app and launches uvicorn on port 8001.
# ──────────────────────────────────────────────────────────────────────

from tools.raive_feeder.main import main

main()
