---
title: MediBot - AI Symptom Checker
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
---

# 🏥 MediBot: AI-Powered Symptom Checker

An intelligent medical symptom checker powered by a **Multi-Agent ReAct system**, **FAISS vector search**, and **OpenAI GPT-4o-mini** — with support for **text**, **voice**, and **image-based** input.

> **Disclaimer:** MediBot is an AI educational tool. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a healthcare provider.

---

## Features

| Input Mode | How It Works | Technology |
|---|---|---|
| **💬 Text Chat** | Type symptoms in natural language → get diagnosis, severity, precautions | ReAct Agent + FAISS RAG |
| **🎙️ Voice Input** | Record symptoms via microphone → auto-transcribed → same AI pipeline | OpenAI Whisper API |
| **📷 Skin Analysis** | Upload skin condition photo → visual AI analysis + severity scoring | GPT-4o-mini Vision |

### Core Capabilities

- **Multi-Agent RAG System** — 4 specialized tools (diagnosis, severity, description, precautions) backed by FAISS vector search over 256 medical documents
- **ReAct Reasoning** — AI autonomously selects the right tool(s) for each query using Thought → Action → Observation loops
- **Conversation Memory** — Remembers previous symptoms across multi-turn interactions
- **Severity Scoring** — Quantitative assessment (1-7 scale per symptom) with visual severity bars
- **Two-Stage Image Pipeline** — Vision AI analyzes the photo, then feeds text back to the ReAct agent for severity + precautions from the medical dataset
- **Edge Case Handling** — Graceful responses for vague, ambiguous, or invalid inputs

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│               USER INTERFACE (Gradio / Streamlit)     │
│      💬 Text Chat    🎙️ Voice Input    📷 Skin Image  │
└──────────┬──────────────┬──────────────┬─────────────┘
           │              │              │
           │         ┌────┴────┐    ┌────┴────────────┐
           │         │ Whisper │    │ GPT-4o-mini     │
           │         │ API     │    │ Vision          │
           │         │ (STT)   │    │ (Image → Text)  │
           │         └────┬────┘    └────┬────────────┘
           │              │              │
           ▼              ▼              ▼
┌──────────────────────────────────────────────────────┐
│                  ReAct AGENT (LangGraph)               │
│                                                        │
│   GPT-4o-mini LLM  ←→  MemorySaver (per-thread)      │
│                                                        │
│   Tools:                                               │
│   ┌──────────────┐ ┌──────────────┐                   │
│   │ diagnose_    │ │ assess_      │                   │
│   │ disease      │ │ severity     │                   │
│   ├──────────────┤ ├──────────────┤                   │
│   │ describe_    │ │ suggest_     │                   │
│   │ disease      │ │ precautions  │                   │
│   └──────────────┘ └──────────────┘                   │
│           ↕               ↕                            │
│   ┌────────────────────────────────────┐              │
│   │    FAISS Vector Store (256 docs)    │              │
│   │    all-MiniLM-L6-v2 embeddings     │              │
│   └────────────────────────────────────┘              │
└──────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| LLM | OpenAI GPT-4o-mini | Language understanding & generation |
| Agent Framework | LangGraph (ReAct) | Autonomous tool selection & reasoning |
| Vector Store | FAISS | Semantic search over medical data |
| Embeddings | all-MiniLM-L6-v2 | 384-dim text embeddings |
| Speech-to-Text | OpenAI Whisper API | Voice input transcription |
| Vision AI | GPT-4o-mini (multimodal) | Skin condition image analysis |
| UI (Option 1) | Gradio | Tabbed web interface |
| UI (Option 2) | Streamlit | Alternative web interface |
| Memory | LangGraph MemorySaver | Per-thread conversation history |
| Datasets | 4 CSV files (5,135 rows) | Medical knowledge base |

---

## Project Structure

```
Medibot/
├── MediBot.ipynb              # Main notebook (9 milestones, 46 steps)
├── MediBot_Colab.ipynb        # Google Colab version
├── app.py                     # Gradio standalone deployment
├── streamlit_app.py           # Streamlit standalone deployment
├── requirements.txt           # Python dependencies
├── DataSet/
│   ├── dataset.csv            # Disease-symptom mappings (4,920 rows)
│   ├── Symptom-severity.csv   # Symptom severity weights (133 rows)
│   ├── symptom_Description.csv# Disease descriptions (41 rows)
│   └── symptom_precaution.csv # Precautionary measures (41 rows)
├── faiss_medibot_index/
│   ├── index.faiss            # Saved FAISS vector index
│   └── index.pkl              # Index metadata
├── Medibot_ProblemStatement.docx
└── Evaluation Rubrics - Medibot.xlsx
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Clone & Install

```bash
git clone https://github.com/primishra1987-fse/Medibot.git
cd Medibot
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or on Windows:
```cmd
set OPENAI_API_KEY=sk-your-key-here
```

### 3. Run

**Option A — Gradio:**
```bash
python app.py
```
Opens at `http://localhost:7860` with a public share link.

**Option B — Streamlit:**
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`.

**Option C — Jupyter Notebook:**
Open `MediBot.ipynb` and run all cells sequentially.

---

## Deployment

### Deploy to Hugging Face Spaces (Gradio)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - Select **Gradio** as the SDK
2. Upload these files to the Space:
   - `app.py`
   - `requirements.txt`
   - `DataSet/` folder (all 4 CSVs)
   - `faiss_medibot_index/` folder
3. Add your `OPENAI_API_KEY` as a Space Secret:
   - Settings → Repository secrets → Add `OPENAI_API_KEY`
4. The Space will auto-build and deploy

### Deploy to Streamlit Cloud

1. Push this repo to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → select `streamlit_app.py` as the main file
4. Add your `OPENAI_API_KEY` in Streamlit's Secrets:
   - Advanced settings → Secrets → Add:
     ```toml
     OPENAI_API_KEY = "sk-your-key-here"
     ```
5. Deploy

### Deploy to Google Colab

1. Open `MediBot_Colab.ipynb` in Google Colab
2. Upload the `DataSet/` folder when prompted
3. Run all cells sequentially
4. Enter your OpenAI API key when prompted

---

## Notebook Milestones

| # | Milestone | Steps | Description |
|---|-----------|-------|-------------|
| 1 | Research & Problem Definition | 1-8 | Data loading, EDA setup, scope definition |
| 2 | Dataset Preparation & FAISS | 9-25 | Cleaning, EDA (8+ visualizations), FAISS index |
| 3 | Multi-Agent System | 26-33 | 4 @tool agents + ReAct agent with memory |
| 5 | Testing & Optimization | 34-39 | 6 test scenarios (diagnosis, severity, edge cases) |
| 6 | Deployment & UI | 40-41 | Original Gradio ChatInterface |
| 7 | Documentation | — | Architecture, tech stack, rubric alignment |
| 8 | Speech-to-Text | 42 | Whisper API voice input |
| 9 | Skin Image Analysis | 43-46 | Vision AI + two-stage pipeline + tabbed UI |

---

## Example Queries

| Input | Type | What Happens |
|---|---|---|
| "I have itching, skin rash, and nodal skin eruptions" | Text | `diagnose_disease` + `assess_severity` → fungal infection diagnosis |
| "How serious is high fever with vomiting?" | Text | `assess_severity` → severity score with visual bars |
| "What is diabetes?" | Text | `describe_disease` → plain-English explanation |
| "What precautions for malaria?" | Text | `suggest_precautions` → self-care advice |
| *Record: "I have a cough and cold"* | Voice | Whisper transcribes → same pipeline as text |
| *Upload skin photo* | Image | Vision AI identifies condition → agent scores severity |

---

## License

This project is for educational purposes.

---

Built with LangChain, FAISS, OpenAI GPT-4o-mini, Gradio, and Streamlit.
