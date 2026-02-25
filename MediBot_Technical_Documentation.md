# MediBot: Technical Documentation

## AI-Powered Medical Symptom Checker — System Architecture, Agent Interactions & Setup Guide

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Component Details](#2-component-details)
3. [Agent Interactions & ReAct Reasoning](#3-agent-interactions--react-reasoning)
4. [Inference Process — Step-by-Step](#4-inference-process--step-by-step)
5. [Data Pipeline](#5-data-pipeline)
6. [Setup & Installation Guide](#6-setup--installation-guide)
7. [Deployment Guide](#7-deployment-guide)
8. [API Reference](#8-api-reference)
9. [Configuration & Environment Variables](#9-configuration--environment-variables)

---

## 1. System Architecture Overview

MediBot is a multi-modal AI medical assistant that accepts **text**, **voice**, and **image** inputs. It uses a **ReAct (Reasoning + Action)** agent pattern to autonomously select specialized tools for diagnosis, severity assessment, disease description, and precautionary advice.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GRADIO WEB INTERFACE                             │
│         ┌──────────────┬──────────────┬──────────────┐                  │
│         │  Text Chat   │  Voice Input │  Skin Image  │                  │
│         │  (Textbox)   │ (Microphone) │  (Upload)    │                  │
│         └──────┬───────┴──────┬───────┴──────┬───────┘                  │
└────────────────┼──────────────┼──────────────┼──────────────────────────┘
                 │              │              │
          raw text         audio file     image file
                 │              │              │
                 │         ┌────┴────┐    ┌────┴─────────────┐
                 │         │ OpenAI  │    │ GPT-4o-mini      │
                 │         │ Whisper │    │ Vision API       │
                 │         │ (STT)   │    │ (Image→Text)     │
                 │         └────┬────┘    └────┬─────────────┘
                 │         transcribed text  analysis text
                 │              │              │
                 ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   ReAct AGENT (LangGraph)                               │
│                                                                         │
│   GPT-4o-mini LLM  ←→  MemorySaver (per-thread conversation history)  │
│                                                                         │
│   ReAct Loop: THOUGHT → ACTION → OBSERVATION → repeat until answer     │
│                                                                         │
│   Tools:                                                                │
│   ┌─────────────────┐  ┌─────────────────┐                             │
│   │ diagnose_disease│  │ assess_severity │                             │
│   │ (symptoms→      │  │ (symptoms→      │                             │
│   │  disease list)  │  │  severity score)│                             │
│   ├─────────────────┤  ├─────────────────┤                             │
│   │describe_disease │  │suggest_         │                             │
│   │ (name→          │  │ precautions     │                             │
│   │  description)   │  │ (name→advice)   │                             │
│   └────────┬────────┘  └────────┬────────┘                             │
│            └──────────┬─────────┘                                       │
│                       ▼                                                 │
│            ┌─────────────────────────────────┐                         │
│            │    FAISS Vector Store            │                         │
│            │    256 documents, 384-dim        │                         │
│            │    all-MiniLM-L6-v2 embeddings   │                         │
│            └─────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Input Mode | Preprocessing | Agent Input | Output |
|---|---|---|---|
| Text | None (raw text) | User message | Diagnosis + severity + advice |
| Voice | Whisper API → transcription | Transcribed text | Same as text |
| Image | GPT-4o-mini Vision → analysis text | Visual analysis summary | Skin condition + severity + precautions |

---

## 2. Component Details

### 2.1 LLM — OpenAI GPT-4o-mini

- **Model**: `gpt-4o-mini`
- **Temperature**: 0 (deterministic responses)
- **Max Tokens**: 1024
- **Role**: Serves as the "brain" of the ReAct agent — interprets user queries, decides which tools to call, and synthesizes final responses.

### 2.2 Vector Store — FAISS

- **Library**: `faiss-cpu` via `langchain-community`
- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional vectors, normalized)
- **Total Documents**: 256
  - 41 Disease-Symptom mapping documents (from `dataset.csv`)
  - 133 Symptom severity documents (from `Symptom-severity.csv`)
  - 41 Disease description documents (from `symptom_Description.csv`)
  - 41 Disease precaution documents (from `symptom_precaution.csv`)
- **Search Method**: L2 (Euclidean) distance with cosine similarity via normalization
- **Relevance Thresholds**:
  - Disease matching: score < 1.3
  - Severity matching: score < 1.2 (strict), < 0.8 (preferred)

### 2.3 Agent Framework — LangGraph ReAct

- **Library**: `langgraph.prebuilt.create_react_agent`
- **Pattern**: ReAct (Reasoning + Action) — the agent autonomously decides which tool(s) to call
- **Memory**: `MemorySaver` with per-thread conversation history
  - `gradio_text` — text chat conversations
  - `gradio_voice` — voice input conversations
  - `gradio_skin` — skin analysis conversations

### 2.4 Speech-to-Text — OpenAI Whisper

- **Model**: `whisper-1`
- **Input**: Audio file (WAV, MP3, M4A, WebM)
- **Output**: Plain text transcription
- **Min File Size**: 1000 bytes (filters empty recordings)

### 2.5 Vision AI — GPT-4o-mini Multimodal

- **Model**: `gpt-4o-mini` with image_url content blocks
- **Input**: Base64-encoded JPEG (resized to max 1024px)
- **Output**: Structured analysis (conditions, observations, severity, actions, red flags)
- **Pipeline**: Two-stage — Vision analysis → ReAct agent follow-up for severity scoring from medical dataset

### 2.6 UI — Gradio Blocks

- **Framework**: Gradio 5.x with `gr.Blocks` + `gr.Tabs`
- **Theme**: `gr.themes.Soft(primary_hue="blue", secondary_hue="green")`
- **Tabs**: Text Chat | Voice Input | Skin Analysis
- **Chat Format**: `type="messages"` (OpenAI-style `{"role": ..., "content": ...}`)

---

## 3. Agent Interactions & ReAct Reasoning

### 3.1 What is ReAct?

ReAct (Reasoning + Action) is an agent pattern where the LLM interleaves **thinking** with **tool calls** in a loop:

```
User Query → THOUGHT → ACTION → OBSERVATION → THOUGHT → ... → FINAL ANSWER
```

The LLM does NOT answer from its training data. Instead, it:
1. **Thinks** about what information is needed
2. **Acts** by calling a specialized tool
3. **Observes** the tool's output
4. **Repeats** if more information is needed
5. **Answers** once it has enough evidence

### 3.2 Tool Selection Logic

The ReAct agent uses the system prompt and tool docstrings to decide which tool(s) to call:

| User Intent | Tool(s) Called | Example Query |
|---|---|---|
| Symptom → Disease | `diagnose_disease` + `assess_severity` | "I have itching and skin rash" |
| Severity check | `assess_severity` | "How serious is high fever with vomiting?" |
| Disease info | `describe_disease` | "What is diabetes?" |
| Precautions | `suggest_precautions` | "What precautions for malaria?" |
| Complex query | Multiple tools in sequence | "I have headache — what is it, how serious, what should I do?" |

### 3.3 Multi-Tool Interaction Example

For a symptom query like *"I have itching, skin rash, and nodal skin eruptions"*:

```
THOUGHT: The user is describing symptoms. I should diagnose possible diseases
         AND assess severity to give a complete picture.

ACTION:  diagnose_disease("itching, skin rash, nodal skin eruptions")
OBSERVATION: Possible conditions: 1. Fungal infection (matching symptoms: itching,
             skin rash, nodal skin eruptions...)

ACTION:  assess_severity("itching, skin rash, nodal skin eruptions")
OBSERVATION: Overall Severity: MODERATE. Score: 10/21 (Average: 3.3/7)
             - skin_rash: 5/7, itching: 3/7, nodal_skin_eruptions: 2/7

THOUGHT: I now have both diagnosis and severity information. I can provide
         a comprehensive response.

FINAL ANSWER: Based on your symptoms, you may have Fungal Infection...
              Severity: MODERATE (3.3/7)... Please consult a healthcare professional.
```

### 3.4 Agent Memory & Conversation Context

Each input tab maintains a separate conversation thread via `MemorySaver`:

```
Thread: gradio_text
├── Turn 1: User: "I have headache and fever"
│           Agent: [calls diagnose + severity] → "Possible: Malaria, Typhoid..."
├── Turn 2: User: "What precautions should I take?"
│           Agent: [remembers previous diagnosis] → [calls suggest_precautions("Malaria")]
└── Turn 3: User: "Tell me more about typhoid"
            Agent: [calls describe_disease("Typhoid")] → "Typhoid is..."
```

The agent remembers all previous messages in the thread, enabling natural follow-up questions without repeating symptoms.

### 3.5 Two-Stage Image Pipeline (Skin Analysis)

Skin analysis uses a unique two-stage approach because the FAISS database contains text-based medical data, not images:

```
Stage 1: VISION ANALYSIS
┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ Skin Photo  │ →   │ GPT-4o-mini      │ →   │ Text Analysis:       │
│ (uploaded)  │     │ Vision API       │     │ "Possible eczema,    │
│             │     │ (base64 image)   │     │  moderate severity"  │
└─────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                        │
Stage 2: ReAct AGENT FOLLOW-UP                          ▼
┌──────────────────────┐     ┌──────────────┐     ┌──────────────────┐
│ "Based on visual     │ →   │ ReAct Agent  │ →   │ Severity score + │
│  analysis, assess    │     │ (calls       │     │ Precautions from │
│  severity &          │     │  assess +    │     │ medical database │
│  precautions"        │     │  suggest)    │     │                  │
└──────────────────────┘     └──────────────┘     └──────────────────┘
```

This bridges the gap between visual AI and the text-based medical knowledge base.

---

## 4. Inference Process — Step-by-Step

### 4.1 Text Input Inference

```
User types: "I have been experiencing itching, skin rash, and nodal skin eruptions"
│
├─ Step 1: Gradio captures input via gr.Textbox
│
├─ Step 2: text_chat() handler called
│   └─ Appends {"role": "user", "content": message} to chat history
│
├─ Step 3: chat_with_medibot_core(message, thread_id="gradio_text")
│   └─ Creates config with thread_id for MemorySaver
│   └─ Calls gradio_agent.invoke({"messages": [HumanMessage(content=message)]}, config)
│
├─ Step 4: ReAct Agent Processing (LangGraph)
│   ├─ LLM reads system prompt + conversation history + current message
│   ├─ THOUGHT: "User describes symptoms → I need diagnose_disease AND assess_severity"
│   │
│   ├─ ACTION: diagnose_disease("itching, skin rash, nodal skin eruptions")
│   │   ├─ Embeds query using all-MiniLM-L6-v2 → 384-dim vector
│   │   ├─ FAISS similarity_search_with_score(query, k=10)
│   │   ├─ Filters: source == "disease_symptoms" AND score < 1.3
│   │   └─ Returns: "1. Fungal infection — Matching symptoms: itching, skin_rash..."
│   │
│   ├─ OBSERVATION: [diagnose_disease output]
│   │
│   ├─ ACTION: assess_severity("itching, skin rash, nodal skin eruptions")
│   │   ├─ FAISS search with score filtering (< 1.2)
│   │   ├─ Keyword overlap validation (words > 3 chars)
│   │   ├─ Calculates: total score, average, severity level
│   │   └─ Returns: "MODERATE — Score: 10/21 — skin_rash: 5/7, itching: 3/7..."
│   │
│   ├─ OBSERVATION: [assess_severity output]
│   │
│   └─ FINAL ANSWER: Synthesizes diagnosis + severity into formatted markdown response
│
├─ Step 5: Response returned to chat_with_medibot_core()
│   └─ Extracts result["messages"][-1].content
│
├─ Step 6: text_chat() appends {"role": "assistant", "content": response} to history
│
└─ Step 7: Gradio updates gr.Chatbot display with full conversation
```

### 4.2 Voice Input Inference

```
User records audio via microphone
│
├─ Step 1: Gradio gr.Audio captures audio → saves as temp file
│
├─ Step 2: User clicks "Submit Voice Input"
│   └─ voice_chat(audio_filepath, chat_history) called
│
├─ Step 3: PREPROCESSING — Speech-to-Text
│   ├─ Validates file exists and size > 1000 bytes
│   ├─ Opens audio file in binary mode
│   ├─ Sends to OpenAI Whisper API (model="whisper-1")
│   └─ Returns: transcribed text (e.g., "I have a cough and cold")
│
├─ Step 4: Appends user message: "🎙️ [Voice]: I have a cough and cold"
│
├─ Step 5: chat_with_medibot_core(transcript, thread_id="gradio_voice")
│   └─ [Same ReAct agent pipeline as Text Input — Steps 3-5 above]
│
├─ Step 6: Appends assistant response to chat history
│
└─ Step 7: Gradio updates voice_chatbot + clears audio input
```

### 4.3 Image Input Inference (Skin Analysis)

```
User uploads skin condition photo
│
├─ Step 1: Gradio gr.Image captures image → saves as temp file
│
├─ Step 2: User clicks "Analyze Skin Condition"
│   └─ skin_chat(image_path, chat_history) called
│
├─ Step 3: STAGE 1 — Visual Analysis (GPT-4o-mini Vision)
│   ├─ encode_image_to_base64(image_path)
│   │   ├─ Opens image with PIL
│   │   ├─ Converts RGBA/P → RGB
│   │   ├─ Resizes to max 1024x1024 (preserving aspect ratio)
│   │   └─ Encodes as JPEG base64 string
│   │
│   ├─ Sends to OpenAI Chat API:
│   │   ├─ model: "gpt-4o-mini"
│   │   ├─ content: [text prompt + image_url with base64 data]
│   │   └─ Prompt asks for: conditions, observations, severity, actions, red flags
│   │
│   └─ Returns: structured text analysis of the skin condition
│
├─ Step 4: STAGE 2 — ReAct Agent Follow-up
│   ├─ Constructs follow-up message:
│   │   "Based on visual analysis: [Stage 1 output]...
│   │    Please assess severity and suggest precautions."
│   │
│   ├─ gradio_agent.invoke(follow_up, thread_id="gradio_skin")
│   │   └─ ReAct agent calls assess_severity + suggest_precautions
│   │       using the identified condition names from Stage 1
│   │
│   └─ Returns: severity scores + precautionary measures from FAISS database
│
├─ Step 5: Combines Stage 1 + Stage 2 into formatted response:
│   ├─ "## 📷 Visual Skin Analysis" — [Stage 1 output]
│   └─ "## 🩺 Severity & Precautions" — [Stage 2 output]
│
└─ Step 6: Gradio updates skin_chatbot + clears image input
```

---

## 5. Data Pipeline

### 5.1 Dataset Overview

| File | Rows | Content | Usage |
|---|---|---|---|
| `dataset.csv` | 4,920 | Disease → Symptom_1 ... Symptom_17 | Build disease-symptom mappings |
| `Symptom-severity.csv` | 133 | Symptom → weight (1-7) | Severity scoring |
| `symptom_Description.csv` | 41 | Disease → Description | Disease explanations |
| `symptom_precaution.csv` | 41 | Disease → Precaution_1 ... Precaution_4 | Self-care advice |

### 5.2 Document Construction

Raw CSV data is transformed into 256 natural-language documents for FAISS indexing:

```
dataset.csv (4,920 rows) → Deduplicated to 41 unique diseases
    → "Disease: Fungal infection. Symptoms: itching, nodal_skin_eruptions, skin_rash"

Symptom-severity.csv (133 rows) → 133 documents
    → "Symptom: skin_rash. Severity weight: 5 out of 7."

symptom_Description.csv (41 rows) → 41 documents
    → "Disease: Fungal infection. Description: Fungal infection is..."

symptom_precaution.csv (41 rows) → 41 documents
    → "Disease: Fungal infection. Precautions: bath twice; use detol; keep infected area dry"
```

### 5.3 Retrieval Strategy

Each tool filters FAISS results by **source metadata** and **relevance score**:

```python
# diagnose_disease: finds matching diseases
results = vectorstore.similarity_search_with_score(symptoms, k=10)
filtered = [(doc, score) for doc, score in results
            if doc.metadata["source"] == "disease_symptoms" and score < 1.3]

# assess_severity: finds matching symptom weights
filtered = [(doc, score) for doc, score in results
            if doc.metadata["source"] == "severity" and score < 1.2]
# + keyword overlap validation to eliminate false positives
```

This source-filtered retrieval ensures each tool only accesses its relevant document subset.

---

## 6. Setup & Installation Guide

### 6.1 Prerequisites

- **Python**: 3.10 or higher
- **OpenAI API Key**: [Get one here](https://platform.openai.com/api-keys) — requires GPT-4o-mini access
- **Disk Space**: ~500 MB (for sentence-transformers model download)
- **RAM**: 2 GB minimum

### 6.2 Installation

```bash
# 1. Clone the repository
git clone https://github.com/primishra1987-fse/Medibot.git
cd Medibot

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### 6.3 Dependencies

| Package | Purpose |
|---|---|
| `langchain`, `langchain-core` | Agent framework, tools, documents |
| `langchain-openai` | OpenAI LLM integration |
| `langchain-community` | FAISS vector store wrapper |
| `langchain-huggingface` | HuggingFace embeddings |
| `langgraph` | ReAct agent with memory |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | all-MiniLM-L6-v2 embedding model |
| `gradio` | Web UI framework |
| `openai` | Whisper API + Vision API |
| `pandas`, `numpy` | Data processing |
| `Pillow` | Image processing |

### 6.4 Setting the API Key

```bash
# Linux / Mac
export OPENAI_API_KEY="sk-your-key-here"

# Windows Command Prompt
set OPENAI_API_KEY=sk-your-key-here

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-your-key-here"
```

### 6.5 Running the Application

**Option A — Gradio (Recommended):**
```bash
python app.py
```
Opens at `http://localhost:7860` with a public share link.

**Option B — Jupyter Notebook:**
Open `MediBot.ipynb` and run all cells sequentially (Step 1 through Step 46).

**Option C — Google Colab:**
Open `MediBot_Colab.ipynb` in Google Colab, upload the `DataSet/` folder, and run all cells.

---

## 7. Deployment Guide

### 7.1 Hugging Face Spaces (Gradio)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) — select **Gradio** SDK
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `DataSet/` folder (all 4 CSVs)
   - `faiss_medibot_index/` folder
3. Add `OPENAI_API_KEY` as a Space Secret: Settings → Repository secrets
4. The Space auto-builds and deploys

**Live Demo**: [huggingface.co/spaces/primishra1987/medibotAssistant](https://huggingface.co/spaces/primishra1987/medibotAssistant)

### 7.2 Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `streamlit_app.py`
4. Add secret: `OPENAI_API_KEY = "sk-your-key-here"`

### 7.3 Docker (Self-Hosted)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV OPENAI_API_KEY=""
EXPOSE 7860
CMD ["python", "app.py"]
```

---

## 8. API Reference

### 8.1 Tool Functions

#### `diagnose_disease(symptoms: str) -> str`
- **Input**: Comma-separated symptom description
- **Process**: FAISS search → filter by `source="disease_symptoms"` and `score < 1.3`
- **Output**: Top 5 matching diseases with symptom overlap

#### `assess_severity(symptoms: str) -> str`
- **Input**: Comma-separated symptom description
- **Process**: FAISS search → filter by `source="severity"` and `score < 1.2` → keyword validation → severity calculation
- **Output**: Severity level (MILD/MODERATE/HIGH/SEVERE), score breakdown, visual bars

#### `describe_disease(disease_name: str) -> str`
- **Input**: Disease name
- **Process**: FAISS search → filter by `source="description"`
- **Output**: Disease explanation + related conditions

#### `suggest_precautions(disease_name: str) -> str`
- **Input**: Disease name
- **Process**: FAISS search → filter by `source="precaution"`
- **Output**: Precautionary measures + general health tips

### 8.2 Handler Functions

| Function | Trigger | Inputs | Outputs |
|---|---|---|---|
| `text_chat(message, history)` | Send button / Enter key | text + chat history | updated history |
| `voice_chat(audio, history)` | Submit Voice button | audio filepath + history | updated history |
| `skin_chat(image, history)` | Analyze button | image filepath + history | updated history |
| `transcribe_audio(filepath)` | Called by voice_chat | audio file path | transcribed text |
| `analyze_skin_condition(path)` | Called by skin_chat | image file path | analysis text |

---

## 9. Configuration & Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o-mini + Whisper |
| `GRADIO_SSR_MODE` | Auto-set | Set to `"false"` in app.py for HF Spaces compatibility |

### Tunable Parameters (in `app.py`)

| Parameter | Default | Location | Purpose |
|---|---|---|---|
| LLM temperature | 0 | `ChatOpenAI(temperature=0)` | Response determinism |
| LLM max_tokens | 1024 | `ChatOpenAI(max_tokens=1024)` | Response length limit |
| Disease score threshold | 1.3 | `diagnose_disease()` | FAISS relevance cutoff |
| Severity score threshold | 1.2 | `assess_severity()` | Severity relevance cutoff |
| Keyword score threshold | 0.8 | `assess_severity()` | Preferred match threshold |
| Image max size | 1024px | `encode_image_to_base64()` | Max image dimension |
| Audio min size | 1000 bytes | `transcribe_audio()` | Empty recording filter |

---

*Document Version: 1.0 | Project: MediBot AI Symptom Checker | Date: February 2026*
