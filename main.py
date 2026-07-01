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

# ── Embedding model ────────────────────────────────────────────
EMBEDDING_MODEL  = 'intfloat/multilingual-e5-base'
COLLECTION_NAME  = 'bellevie_knowledge_multilingual'
TOP_K_RETRIEVAL  = 5

# ── Model cascade ─────────────────────────────────────────────
LLM_MODEL     = 'llama-3.3-70b-versatile'
LLM_FALLBACKS = [
    'openai/gpt-oss-20b',
]
TRANSLATE_MODEL = 'openai/gpt-oss-20b'
OPENAI_FALLBACK_MODEL = 'gpt-4o-mini'

# ── Query cache ───────────────────────────────────────────────
QUERY_CACHE: Dict[str, str] = {}
CACHE_MAX_SIZE = 50

# ── Load on startup ───────────────────────────────────────────
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print(f"Connecting to ChromaDB collection: {COLLECTION_NAME}...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
except Exception as e:
    print(f"❌ Could not open collection '{COLLECTION_NAME}': {e}")
    print(f"   Did you run migrate_to_multilingual_embeddings.py yet?")
    raise

groq_client = Groq(api_key=GROQ_API_KEY)

openai_client = None
if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ OpenAI fallback (gpt-4o-mini) enabled")
else:
    print("ℹ️  No OPENAI_API_KEY found — OpenAI fallback disabled")

print(f"✅ Ready — {collection.count()} vectors loaded (multilingual embeddings)")

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful and empathetic healthcare assistant for BelleVie Global Health Services — a comprehensive healthcare facilitation company based in Dhaka, Bangladesh. Your tagline is "A friend in need" ("প্রয়োজনে একজন বন্ধু").

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
- BelleVie Health Protection System (বেলভি স্বাস্থ্য সুরক্ষা ব্যবস্থা): A special program for NGO workers and loan recipients in Bangladesh offering 12 layers of health protection, digital health management, telemedicine, critical illness coverage, international medical support, and community health management.

RULES YOU MUST STRICTLY FOLLOW:
1. Only answer based on the context provided. Do not make up doctors, hospitals, or services not in the context.
2. NEVER give specific cost or price estimates for medical treatments, surgeries, or hospital stays. If asked about cost, always say: "Treatment costs vary depending on your specific condition and requirements. Please contact BelleVie at 01805464400 or email info.belleviebd@gmail.com for a personalized cost estimate." (In Bangla: "চিকিৎসার খরচ আপনার নির্দিষ্ট অবস্থা ও প্রয়োজনের উপর নির্ভর করে ভিন্ন হতে পারে। সঠিক খরচ জানতে অনুগ্রহ করে বেলভির সাথে যোগাযোগ করুন 01805464400 অথবা ইমেইল করুন info.belleviebd@gmail.com।")
3. If the context does not contain enough information to answer, say: "I don't have that information right now. Please contact BelleVie directly at 01805464400 or email info.belleviebd@gmail.com." (In Bangla: "এই মুহূর্তে আমার কাছে এই তথ্যটি নেই। অনুগ্রহ করে সরাসরি বেলভির সাথে যোগাযোগ করুন 01805464400 অথবা ইমেইল করুন info.belleviebd@gmail.com।")
4. When a user asks about a doctor's appointment time or schedule, always respond with: "To book an appointment, please call BelleVie at 01805464400 or email info.belleviebd@gmail.com. Our team will arrange everything for you." (In Bangla: "অ্যাপয়েন্টমেন্ট বুক করতে অনুগ্রহ করে বেলভিতে কল করুন 01805464400 অথবা ইমেইল করুন info.belleviebd@gmail.com। আমাদের টিম সবকিছু ব্যবস্থা করে দেবে।")
5. Always be warm, clear, and patient. Users may be worried about their health.
6. When recommending hospitals or doctors, briefly explain why they are relevant to the user's condition.
7. Always end responses about overseas treatment by reminding the user that BelleVie handles visa, airport reception, and translation support.
8. Keep responses concise and easy to understand. Avoid overly technical language.
9. If a user seems to be in a medical emergency, immediately direct them to call 01805464400 for BelleVie's 24/7 emergency line.
10. When asked about total number of hospitals or doctors, use the factual numbers above — do not guess from context.
11. When asked what specialist doctors are available in Bangladesh, list the specialties from the factual information above clearly.
12. If we do not have a specific specialist in Bangladesh, honestly say so and suggest the overseas option or advise calling BelleVie.
13. CRITICAL — LANGUAGE RULE: You MUST detect the language of the user's message and respond in that EXACT same language. If the user writes in English, respond in English ONLY. If the user writes in Bangla script, respond in Bangla ONLY. If the user writes in Banglish, respond in Banglish. NEVER switch languages unless the user switches first.
    (ভাষার নিয়ম: ব্যবহারকারী যে ভাষায় বার্তা লেখেন বা বলেন, আপনাকে অবশ্যই সেই একই ভাষায় উত্তর দিতে হবে। ব্যবহারকারী বাংলায় লিখলে শুধু বাংলায় উত্তর দিন। ব্যবহারকারী ইংরেজিতে লিখলে শুধু ইংরেজিতে উত্তর দিন। ব্যবহারকারী নিজে ভাষা পরিবর্তন না করলে আপনি কখনোই ভাষা পরিবর্তন করবেন না।)
    Use formal, respectful আপনি (not তুমি) when addressing the user in Bangla — this is a healthcare context and formal address is expected regardless of the user's age or how they address you.
14. SPELLING & GRAMMAR: Ensure absolutely no spelling mistakes or grammatical errors in your responses. For Bengali (Bangla), strictly follow standard Bangla spelling rules (প্রমিত বাংলা বানানের নিয়ম) — avoid informal transliteration, avoid mixing English words into a Bangla sentence unless the term has no natural Bangla equivalent (e.g. proper nouns like hospital names, "BelleVie", or medical specialty names where the Bangla term would be unfamiliar to patients). For English, ensure proper spelling and grammar.
    (বানান ও ব্যাকরণ: আপনার উত্তরে কোনো বানান বা ব্যাকরণগত ভুল থাকা চলবে না। বাংলার ক্ষেত্রে প্রমিত বাংলা বানানের নিয়ম কঠোরভাবে মেনে চলুন এবং অপ্রয়োজনীয় ইংরেজি শব্দ মেশানো এড়িয়ে চলুন, তবে প্রতিষ্ঠানের নাম, হাসপাতালের নাম বা চিকিৎসা বিশেষজ্ঞের নাম যেগুলোর প্রচলিত বাংলা প্রতিশব্দ রোগীদের কাছে অপরিচিত হতে পারে, সেগুলো ব্যতিক্রম।)
15. NEVER repeat information you have already given in this conversation. If the user asks a follow‑up, build on what was said before rather than starting from scratch.

CONTACT INFORMATION TO SHARE WHEN RELEVANT:
- Phone: 01805464400
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
    prefixed_query = "query: " + query
    vector  = embedder.encode(prefixed_query).tolist()
    results = collection.query(
        query_embeddings=[vector],
        n_results=TOP_K_RETRIEVAL
    )
    return results['documents'][0]

def build_context(docs: list) -> str:
    context = ""
    for i, doc in enumerate(docs):
        trimmed  = doc[:1500] + "..." if len(doc) > 1500 else doc
        context += f"[Source {i+1}]\n{trimmed}\n\n"
    return context.strip()

def get_cache_key(raw_message: str) -> str:
    normalised = raw_message.strip().lower()
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]

def cache_get(raw_message: str) -> Optional[str]:
    return QUERY_CACHE.get(get_cache_key(raw_message))

def cache_set(raw_message: str, response: str) -> None:
    key = get_cache_key(raw_message)
    if key in QUERY_CACHE:
        return
    if len(QUERY_CACHE) >= CACHE_MAX_SIZE:
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
    pre_fetched_context: Optional[str] = None
) -> tuple:
    lang = "Bangla" if not is_english(user_message) else "English"

    # Cache check (text chat only, not voice)
    if not is_voice and len(conversation_history) == 0:
        cached = cache_get(user_message)
        if cached:
            print(f"⚡ Cache hit for: '{user_message[:50]}'")
            conversation_history.append({"role": "user",      "content": user_message})
            conversation_history.append({"role": "assistant", "content": cached})
            return cached, conversation_history, "cache"

    # Retrieve relevant chunks
    if pre_fetched_context is not None:
        context = pre_fetched_context
        print("⚡ Using pre-fetched context")
    else:
        docs    = retrieve(user_message)
        context = build_context(docs)

    # Build messages
    system_instruction = SYSTEM_PROMPT + f"\n\nRELEVANT CONTEXT:\n{context}"

    if is_voice:
        system_instruction += (
            "\n\n[VOICE CALL GUIDELINES: Speak in natural, flowing sentences. "
            "Do NOT cut responses short — always complete your full answer even if it is long. "
            "If explaining a multi‑part topic like the NGO Health Protection System, "
            "cover all parts completely before stopping. "
            "Avoid bullet points — use natural speech instead. "
            "CRITICAL: Avoid grammatical spelling errors in Bangla. "
            "Use standard, formal Bengali (প্রমিত বাংলা) with correct grammar and spelling, "
            "and address the caller formally using আপনি.]"
        )

    messages = [{"role": "system", "content": system_instruction}]
    # Increased history window from -4 to -8 to reduce repetition
    messages += conversation_history[-8:]
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
            "01805464400 or info.belleviebd@gmail.com."
        )
        model_used = "fallback_message"

    # Update history
    conversation_history.append({"role": "user",      "content": user_message})
    conversation_history.append({"role": "assistant", "content": assistant_message})

    # Cache the response for text chat (first-turn queries only)
    if not is_voice and len(conversation_history) == 2:
        cache_set(user_message, assistant_message)

    return assistant_message, conversation_history, model_used

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "service": "BelleVie Health Chatbot API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {
        "status":          "healthy",
        "vectors_loaded":  collection.count(),
        "primary_model":   LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "cache_entries":   len(QUERY_CACHE),
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