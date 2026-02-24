"""
MediBot: AI-Powered Symptom Checker — Streamlit Deployment
===========================================================
Standalone Streamlit app for deployment on Streamlit Cloud or any server.

Features:
  - Text Chat: Type symptoms → ReAct agent diagnosis + severity + precautions
  - Voice Input: Upload audio file → Whisper STT → ReAct agent pipeline
  - Skin Analysis: Image upload → GPT-4o-mini Vision → ReAct agent follow-up

Run locally:
  streamlit run streamlit_app.py

Environment Variables Required:
  OPENAI_API_KEY — Your OpenAI API key
"""

import os
import base64
import io
import warnings
import tempfile

import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MediBot - AI Symptom Checker",
    page_icon="🏥",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════
# INITIALIZE (cached so it only loads once)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading MediBot (building FAISS index)...")
def init_medibot():
    """Load data, build FAISS, create tools and agent. Cached across reruns."""

    DATA_DIR = os.path.join(os.path.dirname(__file__), "DataSet")

    # ── Load datasets ──
    df_main = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
    df_severity = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))
    df_description = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
    df_precaution = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))

    for df in [df_main, df_severity, df_description, df_precaution]:
        df.columns = df.columns.str.strip()

    # ── Build disease-symptom mapping ──
    symptom_cols = [c for c in df_main.columns if c.startswith("Symptom")]
    disease_symptoms = {}
    for _, row in df_main.iterrows():
        disease = str(row["Disease"]).strip()
        symptoms = set()
        for col in symptom_cols:
            val = str(row[col]).strip()
            if val and val.lower() != "nan":
                symptoms.add(val)
        if disease not in disease_symptoms:
            disease_symptoms[disease] = set()
        disease_symptoms[disease].update(symptoms)

    # ── Create documents ──
    documents = []
    for disease, symptoms in disease_symptoms.items():
        documents.append(Document(
            page_content=f"Disease: {disease}. Symptoms: {', '.join(sorted(symptoms))}",
            metadata={"source": "disease_symptoms", "disease": disease},
        ))
    for _, row in df_severity.iterrows():
        documents.append(Document(
            page_content=f"Symptom: {row['Symptom']}. Severity weight: {row['weight']} out of 7.",
            metadata={"source": "severity", "symptom": row["Symptom"], "weight": int(row["weight"])},
        ))
    for _, row in df_description.iterrows():
        documents.append(Document(
            page_content=f"Disease: {row['Disease']}. Description: {row['Description']}",
            metadata={"source": "description", "disease": row["Disease"]},
        ))
    for _, row in df_precaution.iterrows():
        precautions = [str(row[c]) for c in df_precaution.columns[1:] if pd.notna(row[c])]
        documents.append(Document(
            page_content=f"Disease: {row['Disease']}. Precautions: {'; '.join(precautions)}",
            metadata={"source": "precaution", "disease": row["Disease"]},
        ))

    # ── Build FAISS ──
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(documents, embeddings)

    # ── LLM ──
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)

    # ── Tools ──
    @tool
    def diagnose_disease(symptoms: str) -> str:
        """Diagnose possible diseases based on patient symptoms.
        Use this tool when a user describes their symptoms and wants to know
        what disease or condition they might have."""
        results = vectorstore.similarity_search(symptoms, k=10)
        disease_results = [r for r in results if r.metadata.get("source") == "disease_symptoms"]
        if not disease_results:
            return "I couldn't find matching diseases. Please try describing your symptoms differently."
        output = "Based on the symptoms described, here are the possible conditions:\n\n"
        seen = set()
        for i, doc in enumerate(disease_results[:5], 1):
            d = doc.metadata["disease"]
            if d not in seen:
                seen.add(d)
                output += f"{i}. **{d}**\n   Matching symptoms: {doc.page_content}\n\n"
        output += "\n⚠️ Disclaimer: This is an AI-based preliminary assessment. Please consult a healthcare professional."
        return output

    @tool
    def assess_severity(symptoms: str) -> str:
        """Assess the severity of patient symptoms and provide an urgency level.
        Use this tool when a user wants to know how serious their symptoms are."""
        results = vectorstore.similarity_search(symptoms, k=15)
        severity_results = [r for r in results if r.metadata.get("source") == "severity"]
        symptom_weights, seen = [], set()
        for doc in severity_results:
            name = doc.metadata.get("symptom", "")
            weight = doc.metadata.get("weight", 0)
            if name and name not in seen:
                seen.add(name)
                symptom_weights.append((name, weight))
        if not symptom_weights:
            return "I couldn't assess severity. Please list specific symptoms."
        total = sum(w for _, w in symptom_weights)
        avg = total / len(symptom_weights)
        mx = 7 * len(symptom_weights)
        if avg <= 2:
            level, advice = "🟢 MILD", "Monitor symptoms and practice self-care."
        elif avg <= 4:
            level, advice = "🟡 MODERATE", "Consider scheduling a doctor's appointment."
        elif avg <= 5.5:
            level, advice = "🟠 HIGH", "Please see a healthcare provider soon."
        else:
            level, advice = "🔴 SEVERE", "Seek immediate medical attention."
        output = f"**Severity: {level}**\n**Score: {total}/{mx}** (Avg: {avg:.1f}/7)\n\n"
        for s, w in sorted(symptom_weights, key=lambda x: -x[1]):
            output += f"- {s}: {w}/7 [{'█' * w + '░' * (7 - w)}]\n"
        output += f"\n**Recommendation:** {advice}"
        return output

    @tool
    def describe_disease(disease_name: str) -> str:
        """Provide a detailed description and explanation of a specific disease."""
        results = vectorstore.similarity_search(f"Disease description: {disease_name}", k=5)
        desc_results = [r for r in results if r.metadata.get("source") == "description"]
        if not desc_results:
            return f"I don't have detailed information about '{disease_name}'."
        best = desc_results[0]
        output = f"**About {best.metadata['disease']}**\n\n{best.page_content}\n\n"
        if len(desc_results) > 1:
            output += "**Related conditions:**\n"
            for d in desc_results[1:3]:
                output += f"- {d.metadata['disease']}\n"
        return output

    @tool
    def suggest_precautions(disease_name: str) -> str:
        """Suggest precautionary measures and self-care tips for a disease."""
        results = vectorstore.similarity_search(f"Precautions for {disease_name}", k=5)
        prec_results = [r for r in results if r.metadata.get("source") == "precaution"]
        if not prec_results:
            return f"General advice: stay hydrated, rest well, and consult a doctor."
        best = prec_results[0]
        return f"**Precautions for {best.metadata['disease']}**\n\n{best.page_content}\n\n- Stay hydrated\n- Get rest\n- Monitor symptoms\n- See a doctor if symptoms worsen"

    tools_list = [diagnose_disease, assess_severity, describe_disease, suggest_precautions]

    system_prompt = """You are MediBot, an AI-powered medical symptom checker assistant.
You are empathetic, professional, and evidence-based.

You have access to the following specialized tools:
1. **diagnose_disease** - Use when the user describes symptoms and wants to know possible diseases
2. **assess_severity** - Use when the user wants to know how serious their symptoms are
3. **describe_disease** - Use when the user asks 'What is [disease]?' or wants disease information
4. **suggest_precautions** - Use when the user wants advice on what to do for a disease

IMPORTANT GUIDELINES:
- Always use the appropriate tool(s) to answer medical questions — do NOT guess from your training data
- For symptom queries, use BOTH diagnose_disease AND assess_severity to give a complete picture
- Be empathetic and professional in your responses
- Always include a disclaimer that you are an AI and not a replacement for professional medical advice
- Format your responses with clear sections and markdown formatting
"""

    memory = MemorySaver()
    agent = create_react_agent(
        model=llm, tools=tools_list, prompt=system_prompt, checkpointer=memory,
    )

    return llm, agent


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def ask_agent(agent, message: str, thread_id: str) -> str:
    """Send a message to the ReAct agent and return the response."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke({"messages": [HumanMessage(content=message)]}, config)
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {e}"


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using OpenAI Whisper API."""
    if not audio_bytes or len(audio_bytes) < 1000:
        return ""
    api_key = os.environ.get("OPENAI_API_KEY", st.session_state.get("api_key", ""))
    client = OpenAI(api_key=api_key)
    # Use tuple format (filename, bytes, content_type) for reliable format detection
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("recording.wav", audio_bytes, "audio/wav"),
        response_format="text",
    )
    return transcript.strip()


def analyze_skin_image(image_bytes: bytes, llm) -> str:
    """Analyze skin condition image using GPT-4o-mini vision via direct OpenAI client."""
    img = Image.open(io.BytesIO(image_bytes))
    print(f"[VISION-v3] Original: {img.size}, mode={img.mode}")
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    print(f"[VISION-v3] Encoded: {img.size[0]}x{img.size[1]}px, base64={len(b64)} chars")
    assert len(b64) > 100, f"Image encoding failed! base64 length={len(b64)}"

    api_key = os.environ.get("OPENAI_API_KEY", st.session_state.get("api_key", ""))
    print(f"[VISION-v3] API key: {'found' if api_key else 'MISSING'} (len={len(api_key)})")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "You are MediBot, a medical AI assistant with expertise in dermatology. "
                    "Analyze this skin condition image and provide:\n\n"
                    "1. **Possible Conditions** - 2-3 most likely skin conditions with reasoning\n"
                    "2. **Key Visual Observations** - Color, texture, pattern, distribution\n"
                    "3. **Estimated Severity** - Mild / Moderate / Severe\n"
                    "4. **Recommended Actions** - Self-care and whether to see a doctor\n"
                    "5. **When to Seek Immediate Help** - Red flags\n\n"
                    "DISCLAIMER: AI visual assessment for educational purposes only. "
                    "Consult a dermatologist for accurate diagnosis."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"}},
            ],
        }],
        max_tokens=1024,
        temperature=0,
    )
    result = response.choices[0].message.content
    print(f"[VISION-v3] Tokens: {response.usage.total_tokens}, preview: {result[:150]}")
    return result


# ══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════

# ── Header ──
st.markdown("# 🏥 MediBot: AI-Powered Symptom Checker")
st.markdown("**Your intelligent medical assistant powered by ReAct Multi-Agent AI**")
st.info("⚠️ **Disclaimer:** MediBot is an AI educational tool. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a healthcare provider.")

# ── API Key ──
api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("Please enter your OpenAI API key in the sidebar to get started.")
    st.stop()

# ── Load MediBot ──
llm, agent = init_medibot()

# ── Tabs ──
tab_text, tab_voice, tab_skin = st.tabs(["💬 Text Chat", "🎙️ Voice Input", "📷 Skin Analysis"])

# ────────────────────────────────────────
# TAB 1: TEXT CHAT
# ────────────────────────────────────────
with tab_text:
    st.markdown("### Type Your Symptoms")
    st.markdown("Describe your symptoms in natural language, ask about diseases, check severity, or get precautionary advice.")

    # Initialize chat history
    if "text_messages" not in st.session_state:
        st.session_state.text_messages = []

    # Display chat history
    for msg in st.session_state.text_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Describe your symptoms...", key="text_input"):
        st.session_state.text_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("MediBot is analyzing..."):
                response = ask_agent(agent, prompt, thread_id="st_text")
            st.markdown(response)
        st.session_state.text_messages.append({"role": "assistant", "content": response})

# ────────────────────────────────────────
# TAB 2: VOICE INPUT
# ────────────────────────────────────────
with tab_voice:
    st.markdown("### Speak Your Symptoms")
    st.markdown("**Option A — Record:** Click the microphone icon below to record your symptoms directly.")
    st.markdown("**Option B — Upload:** Upload a pre-recorded audio file (WAV, MP3, M4A).")
    st.markdown("*Powered by OpenAI Whisper — supports accents and handles background noise well.*")

    if "voice_messages" not in st.session_state:
        st.session_state.voice_messages = []

    # Display voice chat history
    for msg in st.session_state.voice_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Option A: Microphone recording
    st.markdown("---")
    st.markdown("**Record from microphone:**")
    recorded_audio = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="2x",
        pause_threshold=3.0,
    )

    if recorded_audio and "last_recorded_audio" not in st.session_state:
        st.session_state.last_recorded_audio = None

    if recorded_audio and recorded_audio != st.session_state.get("last_recorded_audio"):
        st.session_state.last_recorded_audio = recorded_audio
        st.audio(recorded_audio, format="audio/wav")

        try:
            with st.spinner("Transcribing recorded audio..."):
                transcript = transcribe_audio(recorded_audio)
        except Exception as e:
            st.error(f"Could not transcribe audio: {e}")
            transcript = ""

        if not transcript.strip():
            st.warning("No speech detected in the recording. Please try again.")
        else:
            st.session_state.voice_messages.append({"role": "user", "content": f"🎙️ [Voice]: {transcript}"})
            with st.chat_message("user"):
                st.markdown(f"🎙️ [Voice]: {transcript}")

            with st.chat_message("assistant"):
                with st.spinner("MediBot is analyzing..."):
                    response = ask_agent(agent, transcript, thread_id="st_voice")
                st.markdown(response)
            st.session_state.voice_messages.append({"role": "assistant", "content": response})

    # Option B: File upload
    st.markdown("---")
    st.markdown("**Or upload an audio file:**")
    audio_file = st.file_uploader("Upload audio file (WAV, MP3, M4A)", type=["wav", "mp3", "m4a", "webm", "ogg"], key="voice_upload")

    if st.button("▶️ Submit Uploaded Audio", type="primary", key="voice_btn"):
        if audio_file is None:
            st.warning("Please upload an audio file first.")
        else:
            audio_bytes = audio_file.read()
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_bytes)

            if not transcript.strip():
                st.warning("No speech detected in the recording. Please try again.")
            else:
                st.session_state.voice_messages.append({"role": "user", "content": f"🎙️ [Voice]: {transcript}"})
                with st.chat_message("user"):
                    st.markdown(f"🎙️ [Voice]: {transcript}")

                with st.chat_message("assistant"):
                    with st.spinner("MediBot is analyzing..."):
                        response = ask_agent(agent, transcript, thread_id="st_voice")
                    st.markdown(response)
                st.session_state.voice_messages.append({"role": "assistant", "content": response})

    if st.button("🗑️ Clear Voice Chat", key="voice_clear"):
        st.session_state.voice_messages = []
        st.rerun()

# ────────────────────────────────────────
# TAB 3: SKIN ANALYSIS
# ────────────────────────────────────────
with tab_skin:
    st.markdown("### Upload a Skin Condition Photo")
    st.markdown("Upload a clear, well-lit photo of the affected area. The AI analyzes visual characteristics and suggests possible conditions.")
    st.markdown("*Powered by GPT-4o-mini Vision AI + ReAct Agent for severity scoring.*")

    if "skin_messages" not in st.session_state:
        st.session_state.skin_messages = []

    # Display skin chat history
    for msg in st.session_state.skin_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    uploaded_image = st.file_uploader("Upload skin condition photo", type=["jpg", "jpeg", "png", "webp"], key="skin_upload")

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded image", width=300)

    if st.button("🔍 Analyze Skin Condition", type="primary", key="skin_btn"):
        if uploaded_image is None:
            st.warning("Please upload an image first.")
        else:
            image_bytes = uploaded_image.read()

            st.session_state.skin_messages.append({"role": "user", "content": "📷 [Uploaded skin condition image]"})

            with st.chat_message("assistant"):
                with st.spinner("Analyzing image (Stage 1: Vision AI)..."):
                    visual_analysis = analyze_skin_image(image_bytes, llm)

                with st.spinner("Getting severity & precautions (Stage 2: ReAct Agent)..."):
                    followup = (
                        f"Based on a visual analysis of a skin condition image:\n\n"
                        f"{visual_analysis}\n\n"
                        f"Please assess the severity and suggest precautions."
                    )
                    agent_resp = ask_agent(agent, followup, thread_id="st_skin")

                full = f"## 📷 Visual Skin Analysis\n\n{visual_analysis}\n\n---\n\n## 🩺 Severity & Precautions\n\n{agent_resp}"
                st.markdown(full)

            st.session_state.skin_messages.append({"role": "assistant", "content": full})

    if st.button("🗑️ Clear Skin Analysis", key="skin_clear"):
        st.session_state.skin_messages = []
        st.rerun()

# ── Footer ──
st.markdown("---")
st.markdown("**MediBot** | LangChain + FAISS + GPT-4o-mini + Streamlit  \n**Features:** 💬 Text | 🎙️ Voice (Whisper) | 📷 Skin Analysis (Vision AI)")
