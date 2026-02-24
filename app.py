"""
MediBot: AI-Powered Symptom Checker — Gradio Deployment
========================================================
Standalone Gradio app for deployment on Hugging Face Spaces or any server.

Features:
  - Text Chat: Type symptoms → ReAct agent diagnosis + severity + precautions
  - Voice Input: Microphone → Whisper STT → ReAct agent pipeline
  - Skin Analysis: Image upload → GPT-4o-mini Vision → ReAct agent follow-up

Environment Variables Required:
  OPENAI_API_KEY — Your OpenAI API key
"""

import os
import base64
import io
import warnings

import pandas as pd
import gradio as gr
from PIL import Image
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

warnings.filterwarnings("ignore")

# ── Configuration ──
DATA_DIR = os.path.join(os.path.dirname(__file__), "DataSet")
FAISS_DIR = os.path.join(os.path.dirname(__file__), "faiss_medibot_index")

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA & BUILD FAISS INDEX
# ══════════════════════════════════════════════════════════════

print("Loading datasets...")
df_main = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
df_severity = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))
df_description = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
df_precaution = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))

# Clean column names
for df in [df_main, df_severity, df_description, df_precaution]:
    df.columns = df.columns.str.strip()

# Build disease-symptom mapping
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

# Create documents for FAISS
documents = []
for disease, symptoms in disease_symptoms.items():
    symptom_text = ", ".join(sorted(symptoms))
    documents.append(Document(
        page_content=f"Disease: {disease}. Symptoms: {symptom_text}",
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

print(f"Created {len(documents)} documents. Building FAISS index...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.from_documents(documents, embeddings)
print(f"FAISS index ready — {vectorstore.index.ntotal} vectors.")

# ══════════════════════════════════════════════════════════════
# 2. INITIALIZE LLM
# ══════════════════════════════════════════════════════════════

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Set it as an environment variable or HF Space secret.")
print(f"OpenAI API key found (length={len(api_key)})")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024, api_key=api_key)

# ══════════════════════════════════════════════════════════════
# 3. DEFINE TOOLS
# ══════════════════════════════════════════════════════════════

@tool
def diagnose_disease(symptoms: str) -> str:
    """Diagnose possible diseases based on patient symptoms.
    Use this tool when a user describes their symptoms and wants to know
    what disease or condition they might have."""
    results = vectorstore.similarity_search_with_score(symptoms, k=10)
    disease_results = [(doc, score) for doc, score in results
                       if doc.metadata.get("source") == "disease_symptoms" and score < 1.3]
    if not disease_results:
        return "I couldn't find matching diseases for those symptoms. Please try describing your symptoms differently."
    output = "Based on the symptoms described, here are the possible conditions:\n\n"
    seen = set()
    for i, (doc, score) in enumerate(disease_results[:5], 1):
        disease = doc.metadata["disease"]
        if disease not in seen:
            seen.add(disease)
            output += f"{i}. **{disease}**\n   Matching symptoms: {doc.page_content}\n\n"
    output += "\n⚠️ Disclaimer: This is an AI-based preliminary assessment. Please consult a healthcare professional for accurate diagnosis."
    return output


@tool
def assess_severity(symptoms: str) -> str:
    """Assess the severity of patient symptoms and provide an urgency level.
    Use this tool when a user wants to know how serious their symptoms are."""
    results = vectorstore.similarity_search_with_score(symptoms, k=10)
    # Filter by source type AND relevance score (lower = more relevant for FAISS L2)
    severity_results = [(doc, score) for doc, score in results
                        if doc.metadata.get("source") == "severity" and score < 1.2]
    symptom_weights = []
    seen = set()
    query_lower = symptoms.lower().replace("_", " ")
    for doc, score in severity_results:
        name = doc.metadata.get("symptom", "")
        weight = doc.metadata.get("weight", 0)
        name_words = name.lower().replace("_", " ")
        # Only include symptoms that are relevant to the query
        has_overlap = any(word in query_lower for word in name_words.split()
                         if len(word) > 3)
        if name and name not in seen and (has_overlap or score < 0.8):
            seen.add(name)
            symptom_weights.append((name, weight))
    if not symptom_weights:
        return "I couldn't assess severity for the described symptoms. Please list specific symptoms."
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
    output = f"**Symptom Severity Assessment**\n\n**Overall Severity: {level}**\n**Score: {total}/{mx}** (Average: {avg:.1f}/7)\n\n**Individual Symptom Breakdown:**\n"
    for s, w in sorted(symptom_weights, key=lambda x: -x[1]):
        output += f"  • {s}: {w}/7 [{'█' * w + '░' * (7 - w)}]\n"
    output += f"\n**Recommendation:** {advice}"
    return output


@tool
def describe_disease(disease_name: str) -> str:
    """Provide a detailed description and explanation of a specific disease.
    Use this tool when a user asks 'What is [disease]?' or wants disease information."""
    results = vectorstore.similarity_search(f"Disease description: {disease_name}", k=5)
    desc_results = [r for r in results if r.metadata.get("source") == "description"]
    if not desc_results:
        return f"I don't have detailed information about '{disease_name}'."
    best = desc_results[0]
    output = f"**About {best.metadata['disease']}**\n\n{best.page_content}\n\n"
    if len(desc_results) > 1:
        output += "**Related conditions:**\n"
        for d in desc_results[1:3]:
            output += f"  • {d.metadata['disease']}\n"
    return output


@tool
def suggest_precautions(disease_name: str) -> str:
    """Suggest precautionary measures and self-care tips for a disease.
    Use this tool when a user wants advice on managing a condition."""
    results = vectorstore.similarity_search(f"Precautions for {disease_name}", k=5)
    prec_results = [r for r in results if r.metadata.get("source") == "precaution"]
    if not prec_results:
        return f"I don't have specific precautions for '{disease_name}'. General advice: stay hydrated, rest well, and consult a doctor."
    best = prec_results[0]
    output = f"**Precautionary Measures for {best.metadata['disease']}**\n\n{best.page_content}\n\n"
    output += "**General Health Tips:**\n  • Stay well-hydrated\n  • Get adequate rest\n  • Monitor your symptoms\n  • Consult a healthcare professional if symptoms worsen\n"
    return output


# ══════════════════════════════════════════════════════════════
# 4. CREATE ReAct AGENT
# ══════════════════════════════════════════════════════════════

tools = [diagnose_disease, assess_severity, describe_disease, suggest_precautions]

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
- If the user's query is unclear, ask clarifying questions about their specific symptoms
- Format your responses with clear sections and markdown formatting
- Remember previous symptoms from the conversation to provide context-aware responses
- When multiple diseases are possible, list them with their matching symptoms
"""

gradio_memory = MemorySaver()
gradio_agent = create_react_agent(
    model=llm, tools=tools, prompt=system_prompt, checkpointer=gradio_memory,
)

# ══════════════════════════════════════════════════════════════
# 5. SPEECH-TO-TEXT (Whisper API)
# ══════════════════════════════════════════════════════════════

def transcribe_audio(audio_filepath: str) -> str:
    """Transcribe audio to text using OpenAI Whisper API."""
    if audio_filepath is None:
        return ""
    file_size = os.path.getsize(audio_filepath)
    print(f"[VOICE] Audio file: {audio_filepath}, size: {file_size} bytes")
    if file_size < 1000:
        print("[VOICE] Audio file too small, likely empty recording")
        return ""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key)
    with open(audio_filepath, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="text",
        )
    print(f"[VOICE] Transcript: {transcript[:100]}")
    return transcript.strip()


# ══════════════════════════════════════════════════════════════
# 6. IMAGE-BASED SKIN ANALYSIS (GPT-4o-mini Vision)
# ══════════════════════════════════════════════════════════════

def encode_image_to_base64(image_path: str, max_size: int = 1024) -> str:
    """Read, resize, and base64-encode an image."""
    print(f"[VISION-v3] Encoding image: {image_path}")
    img = Image.open(image_path)
    print(f"[VISION-v3] Original: {img.size}, mode={img.mode}")
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    print(f"[VISION-v3] Encoded: {img.size[0]}x{img.size[1]}px, base64={len(b64)} chars")
    return b64


def analyze_skin_condition(image_path: str) -> str:
    """Analyze a skin condition image using GPT-4o-mini vision via direct OpenAI client."""
    print(f"[VISION-v3] analyze_skin_condition called with: {image_path}")
    b64 = encode_image_to_base64(image_path)
    assert len(b64) > 100, f"Image encoding failed! base64 length={len(b64)}"
    api_key = os.environ.get("OPENAI_API_KEY", "")
    print(f"[VISION-v3] API key: {'found' if api_key else 'MISSING'} (len={len(api_key)})")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "You are MediBot, a medical AI assistant with expertise in dermatology. "
                    "Analyze this image of a skin condition and provide:\n\n"
                    "1. **Possible Conditions** - 2-3 most likely skin conditions with brief reasoning\n"
                    "2. **Key Visual Observations** - Color, texture, pattern, distribution\n"
                    "3. **Estimated Severity** - Mild / Moderate / Severe\n"
                    "4. **Recommended Actions** - Practical self-care and whether to see a doctor\n"
                    "5. **When to Seek Immediate Help** - Red flags requiring urgent attention\n\n"
                    "DISCLAIMER: This is an AI visual assessment for educational purposes only. "
                    "Always consult a qualified dermatologist for accurate diagnosis."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"}},
            ],
        }],
        max_tokens=1024,
        temperature=0,
    )
    result = response.choices[0].message.content
    print(f"[VISION-v3] Tokens used: {response.usage.total_tokens}")
    print(f"[VISION-v3] Response preview: {result[:150]}")
    return result


def skin_analysis_with_followup(image_path: str, agent, thread_id: str = "gradio_skin") -> str:
    """Two-stage pipeline: vision analysis + ReAct agent follow-up."""
    visual = analyze_skin_condition(image_path)
    followup = (
        f"Based on a visual analysis of a skin condition image, the following was observed:\n\n"
        f"{visual}\n\nPlease assess the severity of the identified symptoms and suggest precautions."
    )
    try:
        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke({"messages": [HumanMessage(content=followup)]}, config)
        agent_resp = result["messages"][-1].content
    except Exception as e:
        agent_resp = f"(Could not retrieve additional assessment: {e})"
    return f"## 📷 Visual Skin Analysis\n\n{visual}\n\n---\n\n## 🩺 Severity & Precautions\n\n{agent_resp}"


# ══════════════════════════════════════════════════════════════
# 7. GRADIO HANDLERS
# ══════════════════════════════════════════════════════════════

def chat_with_medibot_core(message: str, thread_id: str = "gradio_text") -> str:
    try:
        config = {"configurable": {"thread_id": thread_id}}
        result = gradio_agent.invoke({"messages": [HumanMessage(content=message)]}, config)
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {e}"


def text_chat(message, history):
    return chat_with_medibot_core(message, thread_id="gradio_text")


def voice_chat(audio_filepath, chat_history):
    if chat_history is None:
        chat_history = []
    if audio_filepath is None:
        chat_history.append({"role": "assistant", "content": "⚠️ Please record or upload an audio file with your symptoms first."})
        return chat_history, None
    try:
        transcript = transcribe_audio(audio_filepath)
        if not transcript.strip():
            chat_history.append({"role": "assistant", "content": "⚠️ No speech detected. Please try again."})
            return chat_history, None
        chat_history.append({"role": "user", "content": f"🎙️ [Voice]: {transcript}"})
        response = chat_with_medibot_core(transcript, thread_id="gradio_voice")
        chat_history.append({"role": "assistant", "content": response})
        return chat_history, None
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
        return chat_history, None


def skin_chat(image_path, chat_history):
    if chat_history is None:
        chat_history = []
    if image_path is None:
        chat_history.append({"role": "assistant", "content": "⚠️ Please upload a skin condition photo first."})
        return chat_history, None
    try:
        chat_history.append({"role": "user", "content": "📷 [Uploaded skin condition image]"})
        analysis = skin_analysis_with_followup(image_path, gradio_agent, thread_id="gradio_skin")
        chat_history.append({"role": "assistant", "content": analysis})
        return chat_history, None
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
        return chat_history, None


# ══════════════════════════════════════════════════════════════
# 8. BUILD GRADIO UI
# ══════════════════════════════════════════════════════════════

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    title="MediBot - AI Symptom Checker",
) as demo:
    gr.Markdown("""
    # 🏥 MediBot: AI-Powered Symptom Checker
    ### Your intelligent medical assistant powered by ReAct Multi-Agent AI
    **Choose your input method:** Type symptoms, speak into your microphone, or upload a skin condition photo.
    > ⚠️ **Disclaimer:** MediBot is an AI educational tool. It is NOT a substitute for professional medical advice.
    """)

    with gr.Tabs():
        # Tab 1: Text Chat
        with gr.Tab("💬 Text Chat"):
            gr.Markdown("### Type Your Symptoms")
            gr.ChatInterface(
                fn=text_chat,
                examples=[
                    "I have been experiencing itching, skin rash, and nodal skin eruptions",
                    "How serious is having a high fever with vomiting and headache?",
                    "What is diabetes?",
                    "What precautions should I take for malaria?",
                ],
            )

        # Tab 2: Voice Input
        with gr.Tab("🎙️ Voice Input"):
            gr.Markdown("### Speak Your Symptoms\n**Record** via microphone or **upload** an audio file (WAV, MP3, M4A), then click **Submit Voice Input**.\n*Powered by OpenAI Whisper*")
            voice_chatbot = gr.Chatbot(label="Voice Conversation", height=400, type="messages")
            voice_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", format="wav", label="🎙️ Record or Upload Audio")
            with gr.Row():
                voice_submit = gr.Button("▶️ Submit Voice Input", variant="primary", size="lg")
                voice_clear = gr.Button("🗑️ Clear")
            voice_submit.click(fn=voice_chat, inputs=[voice_audio, voice_chatbot], outputs=[voice_chatbot, voice_audio])
            voice_clear.click(fn=lambda: ([], None), inputs=None, outputs=[voice_chatbot, voice_audio])

        # Tab 3: Skin Analysis
        with gr.Tab("📷 Skin Analysis"):
            gr.Markdown("### Upload a Skin Condition Photo\nUpload or capture via webcam. AI analyzes visual characteristics.\n*Powered by GPT-4o-mini Vision + ReAct Agent*")
            skin_chatbot = gr.Chatbot(label="Skin Analysis Results", height=400, type="messages")
            skin_image = gr.Image(type="filepath", label="📷 Upload Photo", sources=["upload", "webcam"])
            with gr.Row():
                skin_submit = gr.Button("🔍 Analyze Skin Condition", variant="primary", size="lg")
                skin_clear = gr.Button("🗑️ Clear")
            skin_submit.click(fn=skin_chat, inputs=[skin_image, skin_chatbot], outputs=[skin_chatbot, skin_image])
            skin_clear.click(fn=lambda: ([], None), inputs=None, outputs=[skin_chatbot, skin_image])

    gr.Markdown("""
    ---
    **MediBot** | LangChain + FAISS + GPT-4o-mini + Gradio
    **Features:** 💬 Text | 🎙️ Voice (Whisper) | 📷 Skin Analysis (Vision AI)
    """)

# ── Launch ──
if __name__ == "__main__":
    demo.launch(share=True, debug=True, ssr_mode=False)
