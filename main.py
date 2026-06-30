import os
import re
import time
import hashlib
import chromadb
from groq import Groq
from openai import OpenAI
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict, Optional
from voice_server import handle_voice_call

# ── Load environment ──────────────────────────────────────────
load_dotenv()

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY']     = 'False'

# ── Config ────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv('GROQ_API_KEY')
CHROMA_DIR      = os.getenv('CHROMA_DIR', './chroma_db')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TOP_K_RETRIEVAL = 3   # reduced from 4 — saves ~500 tokens per call, faster LLM

# ── Model cascade ─────────────────────────────────────────────
# Groq free tier daily token limits (approximate):
#   llama-3.3-70b-versatile : ~14,400 tokens/day  ← primary, best quality
#   mixtral-8x7b-32768      : ~500,000 tokens/day ← best fallback, real safety net
#   openai/gpt-oss-20b      : ~200,000 tokens/day ← last Groq resort
LLM_MODEL     = 'llama-3.3-70b-versatile'
LLM_FALLBACKS = [
    'mixtral-8x7b-32768',
    'openai/gpt-oss-20b',
]
TRANSLATE_MODEL = 'openai/gpt-oss-20b'   # cheap, fast, separate Groq quota

# ── OpenAI fallback (paid, $5 budget) ─────────────────────────
# gpt-4o-mini: best performance-per-token in OpenAI's lineup.
# ~$0.15/1M input tokens — a typical call costs ~$0.0002.
# Only used when ALL Groq models are rate-limited.
OPENAI_FALLBACK_MODEL = 'gpt-4o-mini'

# ── Query cache ───────────────────────────────────────────────
# Stores responses for repeated identical queries.
# Keyed by hash of (english_query). Max 50 entries, then oldest is evicted.
# Only used for text chat (/chat endpoint), NOT for voice (context changes per turn).
QUERY_CACHE: Dict[str, str] = {}
CACHE_MAX_SIZE = 50

# ── Load on startup ───────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection    = chroma_client.get_collection('bellevie_knowledge')
groq_client   = Groq(api_key=GROQ_API_KEY)

# OpenAI client — only initialised if key is present
openai_client = None
if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ OpenAI fallback (gpt-4o-mini) enabled")
else:
    print("ℹ️  No OPENAI_API_KEY found — OpenAI fallback disabled")

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
3. If the context does not contain enough information to answer, say: "I don't have that information right now. Please contact BelleVie directly at +8801805464400 or email info.belleviebd@gmail.com."
4. When a user asks about a doctor's appointment time or schedule, always respond with: "To book an appointment, please call BelleVie at +8801805464400 or email info.belleviebd@gmail.com. Our team will arrange everything for you."
5. Always be warm, clear, and patient. Users may be worried about their health.
6. When recommending hospitals or doctors, briefly explain why they are relevant to the user's condition.
7. Always end responses about overseas treatment by reminding the user that BelleVie handles visa, airport reception, and translation support.
8. Keep responses concise and easy to understand. Avoid overly technical language.
9. If a user seems to be in a medical emergency, immediately direct them to call +8801805-464800 for BelleVie's 24/7 emergency line.
10. When asked about total number of hospitals or doctors, use the factual numbers above — do not guess from context.
11. When asked what specialist doctors are available in Bangladesh, list the specialties from the factual information above clearly.
12. If we do not have a specific specialist in Bangladesh, honestly say so and suggest the overseas option or advise calling BelleVie.
13. CRITICAL — LANGUAGE RULE: You MUST detect the language of the user's message and respond in that EXACT same language. If the user writes in English, respond in English ONLY. If the user writes in Bangla script, respond in Bangla ONLY. If the user writes in Banglish, respond in Banglish. NEVER switch languages unless the user switches first.
14. SPELLING & GRAMMAR: Ensure absolutely no spelling mistakes or grammatical errors in your responses. For Bengali (Bangla), strictly follow standard Bangla spelling rules (প্রমিত বাংলা বানানের নিয়ম). For English, ensure proper spelling and grammar.

CONTACT INFORMATION TO SHARE WHEN RELEVANT:
- Phone: +8801805464400
- Email: info.belleviebd@gmail.com
- Website: www.belleviebd.com
- Address: Crown Park (3rd Floor), House 6/4, Block B, Humayun Road, Mohammadpur, Dhaka-1207
"""

# ── FastAPI ───────────────────────────────────────────────────
app = FastAPI(title="BelleVie Health Chatbot API", version="1.0.0")

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
                {
                    "role": "system",
                    "content": (
                        "You are a professional medical translator. Translate the following Bengali (Bangla) patient query "
                        "into fluent English for a search query. Correct any obvious transcription spelling mistakes, noise, "
                        "or grammar issues during translation so the query is clear. Return ONLY the final English translation, no other text."
                    )
                },
                {"role": "user", "content": text}
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
        n_results=TOP_K_RETRIEVAL   # now 3 instead of 4
    )
    return results['documents'][0]

def build_context(docs: list) -> str:
    context = ""
    for i, doc in enumerate(docs):
        trimmed  = doc[:500] + "..." if len(doc) > 500 else doc
        context += f"[Source {i+1}]\n{trimmed}\n\n"
    return context.strip()

def get_cache_key(english_query: str) -> str:
    """SHA256 hash of the normalised query — used as cache dict key."""
    normalised = english_query.strip().lower()
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]

def cache_get(english_query: str) -> Optional[str]:
    """Return cached response if it exists, else None."""
    return QUERY_CACHE.get(get_cache_key(english_query))

def cache_set(english_query: str, response: str) -> None:
    """Store response in cache. Evict oldest entry when cache is full."""
    key = get_cache_key(english_query)
    if key in QUERY_CACHE:
        return   # already cached, no-op
    if len(QUERY_CACHE) >= CACHE_MAX_SIZE:
        # Evict the first (oldest) key
        oldest = next(iter(QUERY_CACHE))
        del QUERY_CACHE[oldest]
        print(f"🗑️ Cache evicted oldest entry (size was {CACHE_MAX_SIZE})")
    QUERY_CACHE[key] = response
    print(f"💾 Cached query (cache size: {len(QUERY_CACHE)}/{CACHE_MAX_SIZE})")

# ── Shared RAG pipeline ───────────────────────────────────────
def chat_pipeline(
    user_message: str,
    conversation_history: list,
    max_tokens: int = 600,
    is_voice: bool = False,
    pre_fetched_context: Optional[str] = None   # passed in from voice parallel fetch
) -> tuple:
    """
    Core RAG logic shared by /chat and /ws/voice.

    pre_fetched_context: if provided (voice path), skip retrieval entirely —
    the context was already fetched in parallel with STT to save time.

    Returns: (response_text, conversation_history, model_used)
    """

    # Step 1 — Language detection and translation
    lang          = "Bangla" if not is_english(user_message) else "English"
    english_query = translate_to_english(user_message)

    # Step 2 — Cache check (text chat only, not voice)
    # Voice turns are not cached because the same question mid-conversation
    # may need a different answer depending on session history.
    if not is_voice and len(conversation_history) == 0:
        cached = cache_get(english_query)
        if cached:
            print(f"⚡ Cache hit for: '{english_query[:50]}'")
            conversation_history.append({"role": "user",      "content": user_message})
            conversation_history.append({"role": "assistant", "content": cached})
            return cached, conversation_history, "cache"

    # Step 3 — Retrieve relevant chunks
    # If voice_server already fetched context in parallel, skip retrieval here.
    if pre_fetched_context is not None:
        context = pre_fetched_context
        print("⚡ Using pre-fetched context (parallel STT+retrieval)")
    else:
        docs    = retrieve(english_query)
        context = build_context(docs)

    # Step 4 — Build messages
    system_instruction = SYSTEM_PROMPT + f"\n\nRELEVANT CONTEXT:\n{context}"

    if is_voice:
        system_instruction += (
            "\n\n[VOICE CALL GUIDELINES: Keep your response natural, conversational, and concise. "
            "Avoid bullet-point lists — speak in flowing sentences as a real assistant would. "
            "Respond directly to the user's question. "
            "CRITICAL: Avoid grammatical spelling errors in Bangla. "
            "Use standard, formal Bengali (প্রমিত বাংলা) with correct grammar and spelling.]"
        )

    messages = [{"role": "system", "content": system_instruction}]
    messages += conversation_history[-4:]
    messages.append({
        "role":    "user",
        "content": f"[Respond in {lang} only.]\n\n{user_message}"
    })

    assistant_message = None
    model_used        = None

    # ── Try all Groq models first ─────────────────────────────
    all_groq_models = [LLM_MODEL] + LLM_FALLBACKS
    for i, model in enumerate(all_groq_models):
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
                print(f"⚠️ Using Groq fallback: {model}")
            break
        except Exception as e:
            if '429' in str(e):
                print(f"⚠️ Groq {model} rate limited")
            else:
                print(f"⚠️ Groq {model} failed: {e}")
            time.sleep(1)
            continue

    # ── If all Groq failed, try OpenAI gpt-4o-mini ───────────
    # gpt-4o-mini: ~$0.15/1M input tokens, excellent quality,
    # handles Bangla well, far better than any free OpenRouter model.
    if assistant_message is None and openai_client:
        print("⚠️ All Groq models exhausted — trying OpenAI gpt-4o-mini...")
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_FALLBACK_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            assistant_message = response.choices[0].message.content
            model_used        = OPENAI_FALLBACK_MODEL
            print(f"🟢 OpenAI fallback: {OPENAI_FALLBACK_MODEL}")
        except Exception as e:
            print(f"⚠️ OpenAI fallback failed: {e}")

    # ── Final fallback ────────────────────────────────────────
    if assistant_message is None:
        assistant_message = (
            "I'm experiencing high traffic right now. "
            "Please try again in a moment or contact us directly at "
            "+8801805-464800 or info.belleviebd@gmail.com."
        )
        model_used = "fallback_message"

    # Step 5 — Update history
    conversation_history.append({"role": "user",      "content": user_message})
    conversation_history.append({"role": "assistant", "content": assistant_message})

    # Step 6 — Cache the response for text chat (first-turn queries only)
    if not is_voice and len(conversation_history) == 2:
        cache_set(english_query, assistant_message)

    return assistant_message, conversation_history, model_used

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "service": "BelleVie Health Chatbot API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {
        "status":         "healthy",
        "vectors_loaded": collection.count(),
        "primary_model":  LLM_MODEL,
        "cache_entries":  len(QUERY_CACHE),
    }

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = []

    response_text, sessions[request.session_id], actual_model = chat_pipeline(
        request.message,
        sessions[request.session_id],
        max_tokens=600,
        is_voice=False
        # pre_fetched_context not passed — text chat does its own retrieval
    )

    sessions[request.session_id] = sessions[request.session_id][-20:]

    return ChatResponse(
        response=response_text,
        session_id=request.session_id,
        model_used=actual_model
    )

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}

@app.delete("/cache")
def clear_cache():
    """Dev endpoint — flush the query cache."""
    QUERY_CACHE.clear()
    return {"status": "cleared", "message": "Query cache flushed"}

# ── Voice WebSocket ───────────────────────────────────────────
@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    await handle_voice_call(
        websocket=websocket,
        session_id=session_id,
        chat_pipeline=chat_pipeline,
        sessions=sessions
    )