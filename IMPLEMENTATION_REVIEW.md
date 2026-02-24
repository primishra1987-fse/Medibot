# MediBot: Implementation Review

## Document Information

| Field | Value |
|---|---|
| Project | MediBot: AI-Powered Symptom Checker |
| Date | 2026-02-24 |
| Repository | C:\Priyanka\Medibot |
| Primary Author | Priyanka |
| Total Commits | 20 |
| Development Period | 2026-02-23 to 2026-02-24 |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Deployment Targets](#2-deployment-targets)
3. [Three Input Modes](#3-three-input-modes)
4. [Technology Choices and Rationale](#4-technology-choices-and-rationale)
5. [Architecture](#5-architecture)
6. [Issues Encountered and Solutions](#6-issues-encountered-and-solutions)
7. [Files Modified](#7-files-modified)
8. [Git Commit History](#8-git-commit-history)
9. [Current Status and Known Issues](#9-current-status-and-known-issues)
10. [Recommendations](#10-recommendations)

---

## 1. Project Overview

MediBot is an AI-powered medical symptom checker that accepts three forms of input -- text, voice, and images -- and returns diagnoses, severity assessments, disease descriptions, and precautionary advice. It is built on a Multi-Agent ReAct architecture powered by LangGraph, uses FAISS for vector-based semantic search over a curated medical dataset, and relies on OpenAI GPT-4o-mini for language understanding, generation, and image analysis.

### Core Capabilities

- **Multi-Agent RAG System** -- Four specialized tools (diagnosis, severity assessment, disease description, precaution suggestion) backed by FAISS vector search over approximately 256 medical documents derived from four CSV datasets.
- **ReAct Reasoning** -- The LangGraph ReAct agent autonomously selects the appropriate tool(s) for each user query using Thought, Action, and Observation loops. For symptom queries, it typically invokes both `diagnose_disease` and `assess_severity` to provide a comprehensive response.
- **Conversation Memory** -- LangGraph MemorySaver maintains per-thread conversation history so the agent can reference previously mentioned symptoms across multi-turn interactions.
- **Severity Scoring** -- Quantitative assessment on a 1-7 scale per symptom, with visual severity bars and a four-tier urgency classification (Mild, Moderate, High, Severe).
- **Two-Stage Image Pipeline** -- GPT-4o-mini Vision analyzes the uploaded photo, then the resulting text description is fed back into the ReAct agent for severity scoring and precaution retrieval from the medical dataset.
- **Edge Case Handling** -- Graceful responses for vague, ambiguous, or invalid inputs; score thresholds and keyword overlap checks to prevent irrelevant matches.

### Dataset Summary

| File | Rows | Content |
|---|---|---|
| dataset.csv | 4,920 | Disease-to-symptom mappings (17 symptom columns per row) |
| Symptom-severity.csv | 133 | Symptom severity weights (1-7 scale) |
| symptom_Description.csv | 41 | Plain-English disease descriptions |
| symptom_precaution.csv | 41 | 4 precautionary measures per disease |

---

## 2. Deployment Targets

The project supports four deployment targets, each with its own entry point and configuration.

### 2.1 Jupyter Notebook (MediBot.ipynb)

- **Purpose:** Primary development and demonstration environment.
- **Structure:** 9 milestones across 78 cells (Steps 1 through 46).
- **Milestones:**

| # | Milestone | Steps | Description |
|---|---|---|---|
| 1 | Research and Problem Definition | 1-8 | Data loading, EDA setup, scope definition |
| 2 | Dataset Preparation and FAISS | 9-25 | Cleaning, EDA (8+ visualizations), FAISS index creation |
| 3 | Multi-Agent System | 26-33 | Four @tool agents + ReAct agent with memory |
| 5 | Testing and Optimization | 34-39 | Six test scenarios (diagnosis, severity, edge cases) |
| 6 | Deployment and UI | 40-41 | Original Gradio ChatInterface |
| 7 | Documentation | -- | Architecture, tech stack, rubric alignment |
| 8 | Speech-to-Text | 42 | Whisper API voice input via `transcribe_audio()` |
| 9 | Skin Image Analysis | 43-46 | Vision AI + two-stage pipeline + tabbed Gradio UI |

### 2.2 Google Colab (MediBot_Colab.ipynb)

- **Purpose:** Cloud notebook for users without a local Python environment.
- **Differences from local notebook:** Prompts user to upload the DataSet folder; installs dependencies inline; uses Colab-specific file I/O.

### 2.3 Gradio (app.py) -- HuggingFace Spaces

- **Purpose:** Standalone web application deployed to HuggingFace Spaces.
- **Entry point:** `app.py`
- **URL:** Accessible via HuggingFace Spaces with a public share link.
- **Key configuration:**
  - `GRADIO_SSR_MODE=false` environment variable set at module load to disable Gradio 5.x experimental SSR.
  - Explicit `api_key` parameter passed to both `ChatOpenAI` and `OpenAI` clients (required for Spaces where env vars are injected as secrets).
  - FAISS index rebuilt from CSV at startup (not loaded from saved binary) to avoid git LFS issues on HuggingFace.

### 2.4 Streamlit (streamlit_app.py) -- Streamlit Cloud

- **Purpose:** Alternative web application deployed to Streamlit Cloud.
- **Entry point:** `streamlit_app.py`
- **URL:** Accessible via Streamlit Cloud.
- **Key configuration:**
  - `.python-version` file pins Python 3.12 (avoids Python 3.13 `imghdr` removal issue).
  - `runtime.txt` also specifies `python-3.12` for platform compatibility.
  - `streamlit>=1.42.0` in requirements.txt ensures `st.audio_input()` availability.
  - `@st.cache_resource` caches the entire MediBot initialization (FAISS index, LLM, agent) across Streamlit reruns.
  - API key sourced from environment variable or sidebar text input.

---

## 3. Three Input Modes

### 3.1 Text Chat

| Aspect | Detail |
|---|---|
| User Action | Types symptoms or medical questions in natural language |
| Processing | Input sent directly to the ReAct agent |
| Agent Behavior | Selects appropriate tool(s): `diagnose_disease`, `assess_severity`, `describe_disease`, or `suggest_precautions` |
| Output | Markdown-formatted diagnosis, severity score with visual bars, disease descriptions, and/or precautionary advice |
| Thread ID | `gradio_text` (Gradio) / `st_text` (Streamlit) |

**Example queries and expected tool invocations:**

| Query | Tools Used |
|---|---|
| "I have itching, skin rash, and nodal skin eruptions" | `diagnose_disease` + `assess_severity` |
| "How serious is high fever with vomiting?" | `assess_severity` |
| "What is diabetes?" | `describe_disease` |
| "What precautions should I take for malaria?" | `suggest_precautions` |

### 3.2 Voice Input

| Aspect | Detail |
|---|---|
| User Action | Records audio via microphone or uploads an audio file (WAV, MP3, M4A) |
| Processing | Audio sent to OpenAI Whisper API (`whisper-1` model) for transcription, then the transcript is fed into the same ReAct agent pipeline as text |
| Output | Displays transcript prefixed with "[Voice]:", followed by the agent response |
| Thread ID | `gradio_voice` (Gradio) / `st_voice` (Streamlit) |
| Minimum file size | 1,000 bytes (below this threshold, treated as empty recording) |

**Implementation differences by platform:**

| Platform | Audio Source | API Call Format |
|---|---|---|
| Gradio (`app.py`) | `gr.Audio(sources=["microphone", "upload"], type="filepath", format="wav")` | File opened with `open(filepath, "rb")` and passed to Whisper |
| Streamlit (`streamlit_app.py`) | `st.audio_input()` (native built-in widget) | Bytes passed as tuple `("recording.wav", audio_bytes, "audio/wav")` |

### 3.3 Skin Image Analysis

| Aspect | Detail |
|---|---|
| User Action | Uploads a photo of a skin condition (JPG, JPEG, PNG, WEBP) or captures via webcam (Gradio) |
| Stage 1 (Vision) | Image is resized (max 1024px), converted to RGB if RGBA/P/LA, base64-encoded, and sent to GPT-4o-mini via the direct OpenAI client with a structured dermatology prompt |
| Stage 2 (Agent) | Vision analysis text is wrapped in a follow-up prompt and sent to the ReAct agent for severity scoring and precaution retrieval from the medical dataset |
| Output | Two-section response: "Visual Skin Analysis" (from Vision) and "Severity and Precautions" (from ReAct agent) |
| Thread ID | `gradio_skin` (Gradio) / `st_skin` (Streamlit) |

**Vision prompt requests five sections:** Possible Conditions, Key Visual Observations, Estimated Severity, Recommended Actions, and When to Seek Immediate Help.

---

## 4. Technology Choices and Rationale

### 4.1 LLM: OpenAI GPT-4o-mini

| Alternative | Why Not Chosen |
|---|---|
| GPT-4o | Significantly higher cost per token; GPT-4o-mini provides comparable quality for medical Q&A at a fraction of the price; GPT-4o-mini also supports vision (multimodal) |
| GPT-3.5-turbo | Inferior reasoning for multi-tool ReAct workflows; worse at following complex system prompts |
| Local LLMs (Llama, Mistral) | Require GPU resources; harder to deploy on HuggingFace Spaces and Streamlit Cloud; no built-in vision capability |
| Claude (Anthropic) | Would require a different API integration; GPT-4o-mini was chosen for cost-effectiveness with the OpenAI ecosystem (Whisper + Vision + Chat in one provider) |

**Decision rationale:** GPT-4o-mini offers the best trade-off between cost, quality, multimodal capability (text + vision), and ecosystem integration (same provider for Whisper STT).

### 4.2 Speech-to-Text: OpenAI Whisper API

| Alternative | Why Not Chosen |
|---|---|
| Local Whisper (whisper Python package) | Requires ~1-4 GB model download; slow on CPU-only environments (HuggingFace Spaces free tier, Streamlit Cloud); adds significant startup latency |
| SpeechRecognition library (Google/Sphinx) | Lower accuracy for medical terminology; Google Web Speech API has usage limits; Sphinx is offline but very inaccurate |
| Deepgram / AssemblyAI | Additional API provider to manage; adds billing complexity; Whisper quality is sufficient |

**Decision rationale:** The OpenAI Whisper API is already available through the `openai` package (installed as a dependency of `langchain-openai`), requires no additional dependencies or model downloads, handles accents and background noise well, and the `whisper-1` model provides high accuracy for medical terminology.

### 4.3 Vision: Direct OpenAI Client (not LangChain ChatOpenAI)

| Approach | Outcome |
|---|---|
| LangChain `ChatOpenAI` with `HumanMessage(content=[...image_url...])` | **Failed.** LangChain's ChatOpenAI does not reliably forward `image_url` content blocks to the OpenAI API, resulting in the model responding with "I'm unable to analyze images directly" |
| Direct `openai.OpenAI()` client with explicit message construction | **Works.** Full control over the message payload ensures the base64 image is properly included in the API request |

**Decision rationale:** After multiple attempts to make LangChain's ChatOpenAI work with multimodal content blocks, the direct OpenAI client was adopted for vision calls. This is used exclusively for the image analysis stage; the ReAct agent still uses LangChain's ChatOpenAI for text-based tool invocations.

### 4.4 Vector Store: FAISS with all-MiniLM-L6-v2 Embeddings

| Alternative | Why Not Chosen |
|---|---|
| ChromaDB | Heavier dependency; requires persistent storage management; FAISS is simpler for in-memory use |
| Pinecone / Weaviate | Cloud-managed vector databases add external dependency, latency, and cost; the dataset is small enough (~256 documents) for in-memory FAISS |
| OpenAI Embeddings | Additional API cost per embedding; all-MiniLM-L6-v2 runs locally on CPU in milliseconds with no API calls |

**Embedding model details:**

| Property | Value |
|---|---|
| Model | all-MiniLM-L6-v2 |
| Dimensions | 384 |
| Normalization | Enabled (`normalize_embeddings=True`) |
| Device | CPU |
| Index type | FAISS flat (L2 distance) |
| Total vectors | ~256 documents |

### 4.5 Agent Framework: LangGraph ReAct

| Alternative | Why Not Chosen |
|---|---|
| LangChain AgentExecutor | Deprecated in favor of LangGraph; less flexible for custom tool orchestration |
| Custom agent loop | More code to maintain; LangGraph's `create_react_agent` handles the Thought/Action/Observation loop, tool binding, and memory checkpointing out of the box |
| CrewAI / AutoGen | Heavier frameworks designed for multi-agent collaboration; overkill for a single-agent-with-tools architecture |

**Agent configuration:**

| Property | Value |
|---|---|
| Agent type | ReAct (Reasoning + Acting) |
| LLM | GPT-4o-mini (temperature=0, max_tokens=1024) |
| Tools | 4 (`diagnose_disease`, `assess_severity`, `describe_disease`, `suggest_precautions`) |
| Memory | LangGraph MemorySaver (in-memory, per-thread) |

---

## 5. Architecture

### 5.1 System Architecture Diagram

```
+--------------------------------------------------------------+
|                    USER INTERFACE                             |
|                                                              |
|     Gradio (app.py)            Streamlit (streamlit_app.py)  |
|     HuggingFace Spaces         Streamlit Cloud               |
|                                                              |
|  +----------------+  +---------------+  +-----------------+  |
|  | Text Chat Tab  |  | Voice Tab     |  | Skin Image Tab  |  |
|  | (text input)   |  | (audio input) |  | (image upload)  |  |
|  +-------+--------+  +-------+-------+  +--------+--------+  |
+----------|--------------------|-----------------------|-------+
           |                    |                       |
           |               +----v-----+           +----v-----------+
           |               | Whisper  |           | GPT-4o-mini    |
           |               | API      |           | Vision API     |
           |               | (STT)    |           | (direct client)|
           |               +----+-----+           +----+-----------+
           |                    |                       |
           |              text transcript         text analysis
           |                    |                       |
           v                    v                       v
+--------------------------------------------------------------+
|                    ReAct AGENT (LangGraph)                    |
|                                                              |
|   Model: GPT-4o-mini (via LangChain ChatOpenAI)             |
|   Memory: MemorySaver (per-thread conversation history)      |
|                                                              |
|   System Prompt: empathetic, professional, evidence-based    |
|   Behavior: selects 1 or more tools per query               |
|                                                              |
|   +------------------+    +------------------+               |
|   | diagnose_disease |    | assess_severity  |               |
|   | (symptoms ->     |    | (symptoms ->     |               |
|   |  disease list)   |    |  urgency score)  |               |
|   +--------+---------+    +--------+---------+               |
|            |                       |                         |
|   +--------+---------+    +--------+---------+               |
|   | describe_disease |    | suggest_         |               |
|   | (name -> info)   |    | precautions      |               |
|   |                  |    | (name -> advice) |               |
|   +--------+---------+    +--------+---------+               |
|            |                       |                         |
|            v                       v                         |
|   +----------------------------------------------+           |
|   |         FAISS Vector Store                    |           |
|   |         ~256 documents                        |           |
|   |         all-MiniLM-L6-v2 embeddings (384-dim) |           |
|   |                                               |           |
|   |   Sources:                                    |           |
|   |   - disease_symptoms (disease -> symptom map) |           |
|   |   - severity (symptom -> weight 1-7)          |           |
|   |   - description (disease -> text)             |           |
|   |   - precaution (disease -> 4 measures)        |           |
|   +----------------------------------------------+           |
+--------------------------------------------------------------+
```

### 5.2 Data Flow: Text Chat

```
User types symptoms
  --> ReAct Agent receives HumanMessage
  --> Agent decides to call diagnose_disease + assess_severity
  --> Each tool queries FAISS with similarity_search_with_score
  --> Results filtered by source type, relevance score (<1.2/1.3), keyword overlap
  --> Tool returns formatted markdown
  --> Agent synthesizes final response with disclaimer
  --> Response displayed in chat interface
```

### 5.3 Data Flow: Voice Input

```
User records audio / uploads file
  --> Audio sent to OpenAI Whisper API (whisper-1 model)
  --> Transcript returned as plain text
  --> Transcript displayed as user message ("[Voice]: ...")
  --> Transcript fed to ReAct Agent (same as Text Chat flow)
  --> Agent response displayed in chat
```

### 5.4 Data Flow: Skin Image Analysis (Two-Stage Pipeline)

```
User uploads skin condition photo
  --> Stage 1: Image preprocessing
      --> Open with PIL, convert RGBA/P/LA to RGB
      --> Resize to max 1024px (thumbnail)
      --> Base64-encode as JPEG (quality=85)
      --> Assert base64 length > 100
  --> Stage 1: Vision analysis
      --> Send to GPT-4o-mini via direct OpenAI client
      --> Structured dermatology prompt requests 5 sections
      --> Returns text analysis of visual observations
  --> Stage 2: ReAct agent follow-up
      --> Vision text wrapped in follow-up prompt
      --> Sent to ReAct agent as a new HumanMessage
      --> Agent calls assess_severity and/or suggest_precautions
      --> Returns severity scoring + precautions from medical dataset
  --> Combined two-section response displayed
```

---

## 6. Issues Encountered and Solutions

### 6.1 Image Analysis Bug: "I'm unable to analyze images directly"

**Problem:** When using LangChain's `ChatOpenAI` to send multimodal messages containing `image_url` content blocks, the model responded with "I'm unable to analyze images directly" -- indicating the image data was not being forwarded to the OpenAI API.

**Root Cause:** LangChain's `ChatOpenAI` class does not reliably serialize and forward `image_url` content blocks within `HumanMessage` objects. The content list was being converted to a plain text string before reaching the API.

**Fix (commit `d4bebe9`):** Replaced the LangChain ChatOpenAI call with a direct `openai.OpenAI()` client call for vision analysis, constructing the message payload explicitly:

```python
client = OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "...dermatology prompt..."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"}},
        ],
    }],
    max_tokens=1024,
    temperature=0,
)
```

**Additional fixes in subsequent commits:**
- `52060d1`: Added RGBA-to-RGB conversion for PNG images with transparency; added debug logging (`[VISION-v3]` prefix).
- `0784c2d`: Added explicit `api_key` parameter, `assert len(b64) > 100` assertion, and comprehensive diagnostic output for troubleshooting on remote deployments.

### 6.2 Voice Recording Issues

This was one of the most iterative debugging cycles in the project, requiring five commits across both Gradio and Streamlit platforms.

**Gradio issues and fixes:**

| Commit | Problem | Fix |
|---|---|---|
| `3f6bff3` | Audio recorded without explicit format caused Whisper API rejection | Added `format="wav"` to `gr.Audio` component; added `sources=["microphone", "upload"]` for both recording and file upload |

**Streamlit issues and fixes:**

| Commit | Problem | Fix |
|---|---|---|
| `062e9bd` | Voice tab only supported file upload, not microphone recording | Added `audio-recorder-streamlit` third-party component |
| `15d5198` | Whisper API returned `BadRequestError` for recorded audio from Streamlit | Fixed file format by passing tuple `("recording.wav", audio_bytes, "audio/wav")` for proper MIME type detection |
| `71983af` | `audio-recorder-streamlit` had unreliable silence detection (auto-stopped too soon or not at all) | Switched to `streamlit-mic-recorder` component |
| `a35ed9a` | `streamlit-mic-recorder` did not capture audio reliably on all browsers | Switched to native `st.audio_input()` (built-in Streamlit widget, available in Streamlit >= 1.42.0), which provides consistent behavior across browsers |

**Key lesson:** Third-party Streamlit audio recording components (`audio-recorder-streamlit`, `streamlit-mic-recorder`) proved unreliable. The native `st.audio_input()` widget introduced in Streamlit 1.42.0 was the only reliable solution.

### 6.3 FAISS Retrieval Accuracy

**Problem (commit `b6f7b3c`):** The FAISS similarity search was returning irrelevant symptoms. For example, a query about "itchy bumps on hands" might return "nose_rash" or other unrelated symptoms because FAISS L2 distance alone was not sufficient to filter out semantically distant but vectorially close results.

**Fix:** Implemented a three-layer filtering strategy in the `assess_severity` tool:

1. **Score threshold:** Only include results with FAISS L2 distance score < 1.2 (lower is more relevant).
2. **Source type filter:** Only consider documents with the correct `source` metadata (e.g., `"severity"` for severity assessment, `"disease_symptoms"` for diagnosis).
3. **Keyword overlap check:** For severity results, verify that at least one word (length > 3 characters) from the symptom name appears in the user query. Results with very low scores (< 0.8) bypass this check as they are considered highly relevant.

```python
# Score threshold + source filter
severity_results = [(doc, score) for doc, score in results
                    if doc.metadata.get("source") == "severity" and score < 1.2]

# Keyword overlap check
has_overlap = any(word in query_lower for word in name_words.split()
                 if len(word) > 3)
if name and name not in seen and (has_overlap or score < 0.8):
    seen.add(name)
    symptom_weights.append((name, weight))
```

### 6.4 HuggingFace Spaces Deployment Issues

This was the most complex deployment, requiring six commits to resolve platform-specific issues.

#### 6.4.1 SSR Mode Error

**Problem:** Gradio 5.x enables experimental Server-Side Rendering (SSR) by default. On HuggingFace Spaces, this caused a "No API found" error because the SSR mode interfered with the Gradio API endpoint discovery.

**Attempted fixes (in order):**

| Commit | Approach | Outcome |
|---|---|---|
| `abfaa42` | Added `ssr_mode=False` parameter to `demo.launch()` | Did not work; `ssr_mode` is not a valid launch parameter in all Gradio versions |
| `541d8cd` | Set `GRADIO_SSR_MODE=false` environment variable | Worked locally but the env var was being set too late in the module load sequence |
| `7e13536` | Removed SSR env var and pinned Gradio to 4.44.1 (pre-SSR) | Would have worked but introduced version conflicts with other dependencies |
| `e74876a` | Set `os.environ["GRADIO_SSR_MODE"] = "false"` at the very top of `app.py`, before any Gradio imports | **Final fix.** This ensures the env var is set before Gradio's module-level initialization reads it |

**Final solution (line 16 of app.py):**

```python
import os
os.environ["GRADIO_SSR_MODE"] = "false"
```

#### 6.4.2 Python 3.13 Compatibility

**Problem:** HuggingFace Spaces and Streamlit Cloud default to Python 3.13, which removed the `imghdr` and `audioop` standard library modules. These modules are used internally by Gradio and audio processing libraries.

**Fix:**
- Added `runtime.txt` with `python-3.12` (for Streamlit Cloud).
- Added `.python-version` file with `3.12` (for Streamlit Cloud).
- Note: For HuggingFace Spaces, the `audioop-lts` package can be added to requirements.txt if needed.

#### 6.4.3 FAISS Index Binary Files

**Problem:** The pre-built FAISS index files (`index.faiss`, `index.pkl`) are binary files that caused issues with standard `git push` to HuggingFace (file size and LFS requirements).

**Fix:** Instead of pushing the saved FAISS index, `app.py` rebuilds the FAISS index from CSV files at startup. This avoids binary file issues entirely and ensures the index is always consistent with the latest dataset. For HuggingFace Spaces, `HfApi.upload_folder()` was used for initial upload.

#### 6.4.4 API Key Authentication

**Problem (commit `535b6c3`):** On HuggingFace Spaces, environment variables set as Space Secrets are available via `os.environ`, but the `ChatOpenAI` and `OpenAI` clients were not picking them up automatically in all cases.

**Fix:** Explicitly passed the `api_key` parameter to both `ChatOpenAI(api_key=api_key)` and `OpenAI(api_key=api_key)` instead of relying on the `OPENAI_API_KEY` environment variable being auto-detected.

### 6.5 Streamlit Cloud Deployment Issues

#### 6.5.1 Python Version

**Problem (commits `f4810d3`, `7e3cbc0`):** Streamlit Cloud defaulted to Python 3.13, where the `imghdr` module was removed. This caused a runtime error during Streamlit import.

**Fix:** Added `.python-version` file containing `3.12` and pinned `streamlit>=1.42.0` in requirements.txt (which includes the fix for `imghdr` deprecation).

#### 6.5.2 Third-Party Microphone Components

**Problem:** As detailed in Section 6.2, third-party Streamlit components for microphone recording were unreliable.

**Fix:** Used native `st.audio_input()` which was introduced in Streamlit 1.42.0.

---

## 7. Files Modified

### Application Files

| File | Purpose | Lines |
|---|---|---|
| `app.py` | Gradio standalone deployment (HuggingFace Spaces) | 449 |
| `streamlit_app.py` | Streamlit standalone deployment (Streamlit Cloud) | 453 |
| `MediBot.ipynb` | Main Jupyter notebook (9 milestones, 78 cells, Steps 1-46) | -- |
| `MediBot_Colab.ipynb` | Google Colab version of the notebook | -- |

### Configuration Files

| File | Purpose | Content |
|---|---|---|
| `requirements.txt` | Python dependencies | 13 packages: faiss-cpu, langchain, langchain-community, langchain-huggingface, langchain-openai, langgraph, sentence-transformers, gradio, openai, pandas, numpy, Pillow, streamlit>=1.42.0 |
| `runtime.txt` | Python version for Streamlit Cloud | `python-3.12` |
| `.python-version` | Python version pin | `3.12` |
| `.gitignore` | Git ignore rules | Excludes __pycache__, .env, .ipynb_checkpoints, .gradio/, .claude/ |
| `README.md` | Project documentation | Architecture, tech stack, deployment instructions, usage examples |

### Data Files

| File | Purpose |
|---|---|
| `DataSet/dataset.csv` | Disease-symptom mappings (4,920 rows) |
| `DataSet/Symptom-severity.csv` | Symptom severity weights (133 rows, 1-7 scale) |
| `DataSet/symptom_Description.csv` | Disease descriptions (41 rows) |
| `DataSet/symptom_precaution.csv` | Precautionary measures (41 rows, 4 precautions each) |
| `faiss_medibot_index/index.faiss` | Pre-built FAISS vector index |
| `faiss_medibot_index/index.pkl` | FAISS index metadata |

### Reference Documents

| File | Purpose |
|---|---|
| `Medibot_ProblemStatement.docx` | Original problem statement |
| `Evaluation Rubrics - Medibot.xlsx` | 9 grading criteria rubric |

---

## 8. Git Commit History

All 20 commits in chronological order, from initial commit to latest fix.

### Initial Build

| Commit | Date | Message | What Changed |
|---|---|---|---|
| `77d867f` | 2026-02-23 10:46 | MediBot: AI-Powered Symptom Checker with Multi-Modal Input | Initial commit with MediBot.ipynb, Colab notebook, datasets, and FAISS index |
| `7f9107e` | 2026-02-23 10:53 | Add README, Gradio deployment, Streamlit deployment, and requirements | Added app.py, streamlit_app.py, requirements.txt, README.md |

### Image Analysis Fixes (3 commits)

| Commit | Date | Message | What Changed |
|---|---|---|---|
| `d4bebe9` | 2026-02-23 11:01 | Fix image analysis: use OpenAI client directly for vision calls | Replaced LangChain ChatOpenAI with direct OpenAI client for vision |
| `52060d1` | 2026-02-23 11:40 | Fix vision: add RGBA conversion, debug logging for image analysis | Added RGBA-to-RGB conversion, `[VISION-v3]` debug logging |
| `0784c2d` | 2026-02-23 12:42 | Fix vision v3: explicit API key, assertions, diagnostics for image analysis | Added explicit api_key, base64 assertion, comprehensive diagnostics |

### Voice Recording Fixes -- Streamlit (6 commits)

| Commit | Date | Message | What Changed |
|---|---|---|---|
| `82fc4b7` | 2026-02-23 12:45 | Add file upload option to voice tab alongside microphone recording | Added file upload as alternative to microphone in Streamlit |
| `a7fc550` | 2026-02-23 12:47 | Fix Streamlit: explicit API key for Whisper transcription | Passed explicit api_key to OpenAI Whisper client |
| `062e9bd` | 2026-02-23 12:53 | Add microphone recording to Streamlit voice tab | Added audio-recorder-streamlit component |
| `15d5198` | 2026-02-23 21:59 | Fix Whisper BadRequestError for recorded audio in Streamlit | Fixed file format with tuple `("recording.wav", bytes, "audio/wav")` |
| `71983af` | 2026-02-23 22:18 | Switch to streamlit-mic-recorder for reliable voice recording | Replaced audio-recorder-streamlit with streamlit-mic-recorder |
| `a35ed9a` | 2026-02-23 22:28 | Switch to native st.audio_input for reliable voice recording | Replaced third-party components with native st.audio_input() |

### Voice Recording Fixes -- Gradio (1 commit)

| Commit | Date | Message | What Changed |
|---|---|---|---|
| `3f6bff3` | 2026-02-23 22:22 | Fix voice recording: add format=wav, diagnostics, explicit API key | Added format="wav" to gr.Audio, sources=["microphone", "upload"] |

### Python Version and Streamlit Cloud (2 commits)

| Commit | Date | Message | What Changed |
|---|---|---|---|
| `f4810d3` | 2026-02-23 12:58 | Add runtime.txt to pin Python 3.12 for Streamlit Cloud | Added runtime.txt |
| `7e3cbc0` | 2026-02-23 13:07 | Fix Python 3.13 imghdr error: pin Python 3.12 + Streamlit>=1.42 | Added .python-version, pinned streamlit>=1.42.0 |

### FAISS and API Key Fixes (2 commits)

| Commit | Date | Message | What Changed |
|---|---|---|---|
| `535b6c3` | 2026-02-23 23:24 | Fix HF deployment: pass API key explicitly to ChatOpenAI | Explicit api_key parameter on ChatOpenAI constructor |
| `b6f7b3c` | 2026-02-23 23:29 | Fix FAISS retrieval: add relevance scoring to filter irrelevant symptoms | Score threshold (<1.2), source filtering, keyword overlap check |

### HuggingFace SSR Fixes (4 commits)

| Commit | Date | Message | What Changed |
|---|---|---|---|
| `abfaa42` | 2026-02-23 23:36 | Fix HF: disable Gradio SSR mode causing 'No API found' error | First attempt to disable SSR |
| `541d8cd` | 2026-02-24 12:06 | Fix SSR: use env var GRADIO_SSR_MODE=false instead of launch param | Switched to environment variable approach |
| `7e13536` | 2026-02-24 12:13 | Remove SSR env var, pin Gradio 4.44.1 on HF Space | Attempted Gradio version pinning |
| `e74876a` | 2026-02-24 12:28 | Re-add SSR disable env var for Gradio 5.x on HF Spaces | Final fix: env var set before imports |

---

## 9. Current Status and Known Issues

### What Works

| Feature | Gradio (app.py) | Streamlit (streamlit_app.py) | Notebook (MediBot.ipynb) |
|---|---|---|---|
| Text chat with ReAct agent | Yes | Yes | Yes |
| Multi-tool invocation (diagnosis + severity) | Yes | Yes | Yes |
| Disease description lookup | Yes | Yes | Yes |
| Precaution suggestions | Yes | Yes | Yes |
| Conversation memory (per-thread) | Yes | Yes | Yes |
| Voice input (microphone) | Yes | Yes (st.audio_input) | Yes (Gradio in-notebook) |
| Voice input (file upload) | Yes | Via st.audio_input | Yes |
| Skin image analysis (upload) | Yes | Yes | Yes |
| Skin image analysis (webcam) | Yes | No | Yes |
| Two-stage vision pipeline | Yes | Yes | Yes |
| FAISS relevance filtering | Yes | Yes | Yes |
| Severity scoring with visual bars | Yes | Yes | Yes |
| Edge case handling | Yes | Yes | Yes |

### Known Issues and Limitations

1. **Voice recording browser permissions:** On Gradio, microphone access requires the user to grant browser permissions. If the page is served over HTTP (not HTTPS), some browsers will block microphone access entirely. HuggingFace Spaces serves over HTTPS, so this is primarily a local development concern.

2. **FAISS index rebuild time:** Both `app.py` and `streamlit_app.py` rebuild the FAISS index from CSV files at startup rather than loading the saved index. This takes a few seconds but avoids binary file deployment issues. In Streamlit, `@st.cache_resource` ensures this happens only once.

3. **Memory is in-memory only:** The `MemorySaver` checkpointer stores conversation history in RAM. If the server restarts (e.g., HuggingFace Space goes to sleep), all conversation history is lost.

4. **Vision analysis disclaimer:** The GPT-4o-mini vision model may refuse to provide specific medical diagnoses for some images, citing its limitations. The system prompt encourages it to provide observations, but individual responses may vary.

5. **Concurrent users:** The MemorySaver uses fixed thread IDs (`gradio_text`, `gradio_voice`, etc.), meaning all users on the same Gradio deployment share the same conversation thread. For a production system, thread IDs should be generated per-session.

6. **API cost:** Each query involves at least one LLM call (ReAct agent reasoning) plus zero or more tool calls (each involving FAISS search but no additional LLM calls). Voice input adds one Whisper API call. Skin analysis adds one Vision API call plus one agent call. Costs are proportional to usage.

7. **Dataset scope:** The medical dataset covers 41 diseases with 133 symptoms. Queries about conditions not in the dataset will return "no matching diseases found" or generic advice.

---

## 10. Recommendations

### Development Workflow

1. **Restart kernel after code changes.** Jupyter notebooks cache function definitions and objects in memory. After modifying tool functions, FAISS parameters, or the system prompt, restart the kernel and re-run all cells to ensure changes take effect.

2. **Test all three tabs after deployment.** Each input mode (text, voice, image) has its own data flow and failure modes. Verify all three work correctly after any deployment update.

3. **Monitor terminal output for debug prefixes.** The following debug prefixes are printed to the server console:
   - `[VISION-v3]` -- Image encoding, API key status, base64 length, token usage, and response preview for skin analysis calls.
   - `[VOICE]` -- Audio file path, file size, and transcript preview for voice input calls.

### Deployment Best Practices

4. **HuggingFace Spaces:** Ensure `OPENAI_API_KEY` is set as a Space Secret (Settings > Repository Secrets). Verify the Space is not in sleep mode (free tier Spaces sleep after inactivity). After waking, the FAISS index will be rebuilt automatically.

5. **Streamlit Cloud:** Ensure the `.python-version` file specifies `3.12` and `OPENAI_API_KEY` is configured in Streamlit Secrets (Advanced Settings > Secrets > `OPENAI_API_KEY = "sk-..."`).

6. **Local development:** Set the `OPENAI_API_KEY` environment variable before launching. On Windows: `set OPENAI_API_KEY=sk-...`. On Linux/Mac: `export OPENAI_API_KEY=sk-...`.

### Future Improvements

7. **Per-session thread IDs.** Replace fixed thread IDs with UUID-based IDs generated per user session to prevent conversation history from leaking between concurrent users.

8. **Persistent memory.** Replace `MemorySaver` with a persistent backend (e.g., SQLite, Redis) so conversation history survives server restarts.

9. **Expanded dataset.** The current dataset covers 41 diseases. Expanding coverage would improve diagnosis accuracy and reduce "no matching diseases" responses.

10. **Rate limiting.** Add rate limiting to prevent excessive API usage, particularly for the Vision and Whisper endpoints which are more expensive per call.

11. **Loading saved FAISS index.** For faster startup, consider loading the pre-built `faiss_medibot_index/` when available and falling back to CSV-based rebuild only when the index files are missing.

---

## Appendix A: Dependency Versions

The `requirements.txt` specifies the following packages:

```
faiss-cpu
langchain
langchain-community
langchain-huggingface
langchain-openai
langgraph
sentence-transformers
gradio
openai
pandas
numpy
Pillow
streamlit>=1.42.0
```

Key version constraints:
- `streamlit>=1.42.0` is required for `st.audio_input()` support.
- `langchain` v1.2+ is required for `langchain_core.documents.Document` and `langchain_core.tools.tool` imports (the older `langchain.schema` import path is deprecated).
- Python 3.12 is recommended (3.13 removes `imghdr` and `audioop`).

## Appendix B: Environment Variables

| Variable | Required | Where Set | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | OS env, HF Secrets, Streamlit Secrets, or sidebar input | Authentication for GPT-4o-mini, Whisper, and Vision APIs |
| `GRADIO_SSR_MODE` | Gradio only | Set in app.py before imports | Disables Gradio 5.x experimental SSR to prevent "No API found" error |

---

*End of Implementation Review*
