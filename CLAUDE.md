# CLAUDE.md — raiveFlier Project Instructions

> **Scope:** Project-specific rules for the raiveFlier codebase.
> **Last updated:** 2026-02-23

---

## 1. Code Annotation Requirement (Pre-Commit)

**Every file that is created or modified must contain inline code annotations before it is committed.**

### What to Annotate

When writing or modifying code in this repo, add inline comments that explain:

- **Module-level overview** — A comment block at the top of each file explaining the file's role in the architecture, its key classes/functions, and how it connects to other layers.
- **Design patterns** — Identify and explain patterns in use (adapter, observer, facade, strategy, dependency injection, fallback chain, etc.).
- **Data flow** — Explain how data moves through the file: what comes in, what goes out, and what transforms happen.
- **Domain context** — Explain rave culture, music database, citation tier, or electronic music concepts that a junior developer wouldn't know.
- **"Why" over "what"** — Focus on explaining *why* a design choice was made, not restating what the code literally does. Self-evident code doesn't need comments.
- **Architectural connections** — Note which layer this file belongs to and what it depends on (e.g., "This service depends on ILLMProvider and IWebSearchProvider interfaces").

### Comment Style Guide

| Language | Comment Syntax | Overview Block |
|---|---|---|
| Python | `#` inline | `# ─── SECTION NAME ───` block at module top |
| JavaScript | `//` inline | `/* ... */` block at module top |
| CSS | `/* ... */` | `/* ... */` block at file top |
| HTML | `<!-- ... -->` | `<!-- ... -->` block at file top |
| YAML | `#` inline | `#` block at file top |
| Shell | `#` inline | `#` block at file top |
| Dockerfile | `#` inline | `#` block at file top |

### What NOT to Do

- Do not add comments that merely restate the code (e.g., `# increment counter` above `counter += 1`).
- Do not add comments to code you did not write or modify in the current session — only annotate new or changed code.
- Do not break code with comments — verify syntax after annotating.
- Do not add docstrings to functions that already have adequate docstrings unless you are changing the function's behavior.

---

## 2. Project Architecture

raiveFlier is a **layered architecture with adapter pattern** for all external services.

### Layer Hierarchy (top → bottom)

```
Frontend (vanilla JS/CSS/HTML)
    ↓ REST + WebSocket
API Layer (FastAPI routes, schemas, middleware)
    ↓
Pipeline (orchestrator, confirmation gate, progress tracker)
    ↓
Services (researchers, citation, interconnection, Q&A, recommendations)
    ↓
Interfaces (abstract base classes — ILLMProvider, IOCRProvider, etc.)
    ↓
Providers (concrete adapters — OpenAI, Anthropic, Discogs, ChromaDB, etc.)
    ↓
Models (Pydantic v2 frozen data objects)
```

### 5-Phase Pipeline

```
Upload → OCR → Entity Extraction → [User Confirmation] → Research → Interconnection → Output
         Phase 1                     Human-in-the-loop     Phase 2-5
```

### Key Design Principles

- **No vendor lock-in** — All external services abstracted behind interfaces.
- **Dependency injection** — All wiring happens in `src/main.py`; services never create their own dependencies.
- **Immutable state** — Pipeline state uses Pydantic `frozen=True` with `model_copy(update={...})` for transitions.
- **Graceful degradation** — Provider fallback chains (LLM: OpenAI → Anthropic → Ollama; OCR: Vision → EasyOCR → Tesseract).
- **Human-in-the-loop** — Pipeline pauses after entity extraction for user review before research begins.

---

## 3. Tech Stack

| Component | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, Pydantic v2 |
| Frontend | Vanilla JS, CSS, HTML (no framework) |
| LLMs | OpenAI, Anthropic, Ollama (via adapters) |
| OCR | LLM Vision, Tesseract, EasyOCR |
| Vector Store | ChromaDB (RAG) |
| Embeddings | FastEmbed, OpenAI, SentenceTransformers, Nomic |
| Music DBs | Discogs, MusicBrainz, Bandcamp, Beatport |
| Logging | structlog (JSON in prod, console in dev) |
| Deployment | Docker on Render (512MB RAM, persistent /data disk) |
| Testing | pytest |

---

## 4. Deployment Constraints

- **512MB RAM** — EasyOCR and SentenceTransformers excluded from Docker build; FastEmbed used instead.
- **Persistent disk at /data** — SQLite databases (session, feedback, flier history) and ChromaDB data survive container restarts.
- **Single worker** — Uvicorn runs with 1 worker to stay within memory budget.
- **Reference corpus** — Auto-ingested from `data/reference_corpus/` on first boot (idempotent).
