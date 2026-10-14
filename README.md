# FORA AI — Autonomous Multi-Agent Deep Research Engine

**FORA AI** is a production-grade, zero-cost autonomous research control plane built with **LangGraph**, the **Model Context Protocol (MCP)**, **Groq Cloud**, and **FastAPI**, styled in an ultra-modern cyber-security dark aesthetic inspired by **Guardbase**.

---
### 🎬 Introducing FORA AI

A short video introducing FORA AI, its purpose, key features, and the technology behind it.

[![FORA AI: Autonomous Deep Research Engine](https://repoclip.io/api/badge/56d7d13d-d54c-4114-9d0c-8ac032068950)](https://repoclip.io/v/56d7d13d-d54c-4114-9d0c-8ac032068950)

## ⚡ What Makes FORA AI Different?

Instead of relying on a single prompt or paid search APIs, FORA AI deploys a **four-agent cognitive loop** that browses the live web, extracts full article content, subjects findings to an adversarial fact-check audit, and synthesizes a verified executive briefing with zero hallucinations.

```
                         [ Research Query ]
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │      PLANNER AGENT       │ (Decomposes thesis into sub-queries)
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │     RESEARCHER AGENT     │ (DuckDuckGo DDGS + SSRF-safe Scraper)
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │       CRITIC AGENT       │ (Audits facts & citation completeness)
                     └────────────┬─────────────┘
                                  │
                   Incomplete?    ├──► Self-Correction Loop (Back to Planner)
                                  │
                   Approved!      ▼
                     ┌──────────────────────────┐
                     │    SYNTHESIZER AGENT     │ (Briefing + Clickable Markdown Citations)
                     └────────────┬─────────────┘
                                  │
                                  ▼
             ┌──────────────────────────────────────────┐
             │   GROUNDED Q&A ENGINE (Temperature 0.0)  │ (Zero-Hallucination Follow-up)
             └──────────────────────────────────────────┘
```

---

## 🛡️ Production Failsafes & Guarantees

1. **100% Free / Zero-Cost Stack**:
   - **LLM**: Groq Cloud free tier (`qwen/qwen3.8-27b`) running at **`temperature = 0.0`** for absolute factual determinism.
   - **Web Search**: DuckDuckGo live scraper (`ddgs`) requires zero API keys and zero monthly fees.
   - **Persistence**: Thread state saved in local SQLite (`checkpoints.db`) via `AsyncSqliteSaver`.
2. **Infinite Loop Hard Failsafe**:
   - Critic terminates at `MAX_RESEARCH_ITERATIONS = 3` to ensure tasks always conclude rather than burning rate limits.
3. **SSRF & Security Shielding**:
   - Scraper rejects loopback, private, link-local, and cloud metadata addresses (`127.0.0.1`, `localhost`, `169.254.169.254`, `10.0.0.0/8`, `192.168.0.0/16`).
   - Prompt injection defense wraps external web payloads in `<untrusted_source_content>` tags.
4. **Citation Grounding Filter**:
   - Discards any citation URL not physically fetched and verified during that task execution.
5. **Grounded Follow-up Q&A Module**:
   - Exclusive follow-up chat engine answering user queries strictly over verified findings at `temperature = 0.0`. If a fact is unverified, it explicitly admits it rather than speculating.

---

## 🖥️ User Interface & Experience

- **Guardbase Cyber Aesthetics**: Midnight obsidian canvas (`#06070a`), glowing electric cyan (`#00f0ff`) and emerald (`#00ff9d`) telemetry accents, frosted glass cards (`backdrop-blur-xl`), and matrix grid background.
- **Live Cognitive Trace**: Watch each agent execute in real time (`[✓] Planner`, `[✓] Researcher`, `[●] Critic`, `[ ] Synthesizer`) with expandable tool execution stream.
- **Interactive Citations**: Superscript badges (`[1]`, `[2]`) with URL tooltips and sources drawer.
- **One-Click Controls**: Copy raw Markdown, export `.md` file, prompt suggestion chips, and Quick vs. Deep depth toggles.

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone <repo-url>
cd research-engine
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Your Free Groq Key
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your free Groq API key (from [https://console.groq.com](https://console.groq.com)):
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
DEFAULT_MODEL=qwen/qwen3.8-27b
```

### 3. Launch the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **FORA AI Web Interface**: [http://localhost:8000/](http://localhost:8000/)
- **FastAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Probe**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📡 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive FORA AI Guardbase UI |
| `POST` | `/api/v1/research` | Submit a research task (`{"topic": str, "depth": "quick" \| "deep"}`) |
| `GET` | `/api/v1/research/{task_id}` | Poll live agent status, active sub-step, and telemetry logs |
| `GET` | `/api/v1/research/{task_id}/report` | Retrieve final synthesized markdown briefing & citations |
| `POST` | `/api/v1/research/{task_id}/ask` | Zero-hallucination follow-up Q&A (`{"question": str}`) |
| `GET` | `/health` | Liveness & readiness probe |

---

## 🧪 Testing

Run the automated test suite:
```bash
pytest -v
```
