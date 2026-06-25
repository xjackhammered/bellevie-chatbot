import os
import re
import time
import chromadb
from groq import Groq
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict
from voice_server import handle_voice_call

# ── Load environment ──────────────────────────────────────────
load_dotenv()

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY']     = 'False'

# ── Config ────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv('GROQ_API_KEY')
CHROMA_DIR      = os.getenv('CHROMA_DIR', './chroma_db')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TOP_K_RETRIEVAL = 4

LLM_MODEL     = 'llama-3.3-70b-versatile'
LLM_FALLBACKS = [
    'gemma2-9b-it',
    'mixtral-8x7b-32768',
    'llama-3.1-8b-instant',
]
TRANSLATE_MODEL = 'llama-3.1-8b-instant'

# ── Load on startup ───────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection    = chroma_client.get_collection('bellevie_knowledge')
groq_client   = Groq(api_key=GROQ_API_KEY)

print(f"✅ Ready — {collection.count()} vectors loaded")

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful and empathetic healthcare assistant for BelleVie Global Health Services — a comprehensive healthcare facilitation company based in Dhaka, Bangladesh. Your tagline is "A friend in need."

BelleVie helps patients with:
- Finding and consulting specialist doctors in Bangladesh and internationally
- Arranging overseas medical treatment in Thailand, India, China, Malaysia, and Turkey
- Home diagnostics, medicine delivery, home nursing, and medical equipment
- Emergency ambulance and ICU/CCU services (24/7)
- Preventive health packages and health insurance support

FACTUAL INFORMATION YOU MUST KNOW:
- BelleVie collaborates with hospitals in exactly 5 countries: Thailand, India, China, Malaysia, and Turkey
- In Thailand: 9 hospitals (MedPark, Sukhumvit, Nakhonthon, Bangkok Hospital, Bumrungrad, Samitivej, Vejthani, Phyathai 2, Rutnin Eye Hospital)
- In India: 69 hospitals across multiple chains including Apollo, Fortis, Manipal, Wockhardt, Rela, Rainbow Children's, and Surya Hospitals
- In China: 7 hospitals in Kunming, Yunnan
- In Bangladesh: 26 specialist doctors available for consultation
- Total international affiliated hospitals: 85
- Specialist doctors in Bangladesh cover these fields: Urology, Surgical Oncology, Radiotherapy & Oncology, Clinical Oncology, Cancer (Oncology), Periodontology, Oral & Maxillofacial Surgery, Gynaecology & Obstetrics, Gastroenterology, Medicine & Gastroenterology, Internal Medicine, Cardiology, Haematology, General Surgery, Laparoscopy & Breast Surgery
- We do NOT currently have eye specialists or pulmonologists available in Bangladesh. For these, recommend overseas options or advise contacting BelleVie directly.
- BelleVie Health Protection System: A special program for NGO workers and loan recipients in Bangladesh offering 12 layers of health protection, digital health management, telemedicine, critical illness coverage, international medical support, and community health management.

RULES YOU MUST STRICTLY FOLLOW:
1. Only answer based on the context provided. Do not make up doctors, hospitals, or services not in the context.
2. NEVER give specific cost or price estimates for medical treatments, surgeries, or hospital stays. If asked about cost, always say: "Treatment costs vary depending on your specific condition and requirements. Please contact BelleVie at +8801805-464800 or email info.belleviebd@gmail.com for a personalized cost estimate."
3. If the context does not contain enough information to answer, say: "I don't have that information right now. Please contact BelleVie directly at +8801805-464800 or email info.belleviebd@gmail.com."
4. When a user asks about a doctor's appointment time or schedule, always respond with: "To book an appointment, please call BelleVie at +8801805-464800 or email info.belleviebd@gmail.com. Our team will arrange everything for you."
5. Always be warm, clear, and patient. Users may be worried about their health.
6. When recommending hospitals or doctors, briefly explain why they are relevant to the user's condition.
7. Always end responses about overseas treatment by reminding the user that BelleVie handles visa, airport reception, and translation support.
8. Keep responses concise and easy to understand. Avoid overly technical language.
9. If a user seems to be in a medical emergency, immediately direct them to call +8801805-464800 for BelleVie's 24/7 emergency line.
10. When asked about total number of hospitals or doctors, use the factual numbers above — do not guess from context.
11. When asked what specialist doctors are available in Bangladesh, list the specialties from the factual information above clearly.
12. If we do not have a specific specialist in Bangladesh, honestly say so and suggest the overseas option or advise calling BelleVie.
13. CRITICAL — LANGUAGE RULE: You MUST detect the language of the user's message and respond in that EXACT same language. If the user writes in English, respond in English ONLY. If the user writes in Bangla script, respond in Bangla ONLY. If the user writes in Banglish, respond in Banglish. NEVER switch languages unless the user switches first.

CONTACT INFORMATION TO SHARE WHEN RELEVANT:
- Phone: +8801805-464800
- Email: info.belleviebd@gmail.com
- Website: www.belleviebd.com
- Address: Crown Park (3rd Floor), House 6/4, Block B, Humayun Road, Mohammadpur, Dhaka-1207
"""

# ── FastAPI ───────────────────────────────────────────────────
app = FastAPI(
    title="BelleVie Health Chatbot API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    session_id: str

class ChatResponse(BaseModel):
    response:   str
    session_id: str
    model_used: str

# ── Session store ─────────────────────────────────────────────
sessions: Dict[str, List[dict]] = {}

# ── Helpers ───────────────────────────────────────────────────
def is_english(text: str) -> bool:
    bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    total_chars  = len(text.replace(' ', ''))
    if total_chars == 0:
        return True
    return (bangla_chars / total_chars) < 0.3

def translate_to_english(text: str) -> str:
    if is_english(text):
        return text
    try:
        response = groq_client.chat.completions.create(
            model=TRANSLATE_MODEL,
            messages=[
                {"role": "system", "content": "Translate the following Bengali text to English. Return ONLY the translation."},
                {"role": "user",   "content": text}
            ],
            max_tokens=200,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except:
        return text

def retrieve(query: str) -> list:
    vector  = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[vector],
        n_results=TOP_K_RETRIEVAL
    )
    return results['documents'][0]

def build_context(docs: list) -> str:
    context = ""
    for i, doc in enumerate(docs):
        trimmed  = doc[:500] + "..." if len(doc) > 500 else doc
        context += f"[Source {i+1}]\n{trimmed}\n\n"
    return context.strip()

# ── Shared RAG pipeline ───────────────────────────────────────
def chat_pipeline(
    user_message: str,
    conversation_history: list,
    max_tokens: int = 600
) -> tuple:
    """
    Core RAG logic shared by both /chat and /ws/voice.
    max_tokens controls response length:
      - 600 for text chat (full detailed responses)
      - 400 for voice calls (detailed but speakable)
    Returns: (response_text, conversation_history, model_used)
    """

    # Step 1 — Language detection and translation
    lang          = "Bangla" if not is_english(user_message) else "English"
    english_query = translate_to_english(user_message)

    # Step 2 — Retrieve relevant chunks
    docs    = retrieve(english_query)
    context = build_context(docs)

    # Step 3 — Build messages
    # No voice_note — system prompt already handles conciseness (Rule 8)
    messages = [
        {
            "role":    "system",
            "content": SYSTEM_PROMPT + f"\n\nRELEVANT CONTEXT:\n{context}"
        }
    ]
    messages += conversation_history[-4:]
    messages.append({
        "role":    "user",
        "content": f"[Respond in {lang} only.]\n\n{user_message}"
    })

    # Step 4 — Model cascade
    all_models        = [LLM_MODEL] + LLM_FALLBACKS
    assistant_message = None
    model_used        = None

    for i, model in enumerate(all_models):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            assistant_message = response.choices[0].message.content
            model_used        = model
            if i > 0:
                print(f"⚠️ Using fallback model: {model}")
            break

        except Exception as e:
            if '429' in str(e) and i < len(all_models) - 1:
                print(f"⚠️ {model} quota hit — trying next...")
                time.sleep(2)
                continue
            else:
                assistant_message = (
                    "I'm experiencing high traffic right now. "
                    "Please try again in a moment or contact us directly at "
                    "+8801805-464800 or info.belleviebd@gmail.com."
                )
                model_used = "fallback_message"
                break

    # Step 5 — Update history
    conversation_history.append({"role": "user",      "content": user_message})
    conversation_history.append({"role": "assistant", "content": assistant_message})

    return assistant_message, conversation_history, model_used   # ← 3 values now

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":  "running",
        "service": "BelleVie Health Chatbot API",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {
        "status":         "healthy",
        "vectors_loaded": collection.count(),
        "primary_model":  LLM_MODEL
    }

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = []

    response_text, sessions[request.session_id], actual_model = chat_pipeline(
        request.message,
        sessions[request.session_id],
        max_tokens=600
    )

    sessions[request.session_id] = sessions[request.session_id][-20:]

    return ChatResponse(
        response=response_text,
        session_id=request.session_id,
        model_used=actual_model   # now shows real model used
    )

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}

# ── Voice WebSocket ───────────────────────────────────────────
@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for live voice calls.
    Flutter connects here for AI voice conversation.
    URL: ws://66.29.151.40:8010/ws/voice/{session_id}
    """
    await handle_voice_call(
        websocket=websocket,
        session_id=session_id,
        chat_pipeline=chat_pipeline,
        sessions=sessions
    )