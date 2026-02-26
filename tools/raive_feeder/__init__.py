# ─── raiveFeeder — Corpus Ingestion & Database Management GUI ───────────
#
# raiveFeeder is a companion web app for raiveFlier that provides a GUI
# for all corpus ingestion and database management operations.  It adds
# capabilities beyond the CLI: audio transcription, image/scan OCR,
# intelligent web scraping, and support for all common document formats.
#
# Architecture: FastAPI backend on port 8001 + vanilla JS frontend.
# Shares models, interfaces, and providers from raiveFlier's src/ package.
# Talks to the same ChromaDB vector store at data/chromadb/.
#
# Launch: python -m tools.raive_feeder.main
# ────────────────────────────────────────────────────────────────────────
