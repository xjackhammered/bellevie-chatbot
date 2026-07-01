import os
import io
import re
import wave
import asyncio
import json
import edge_tts
import webrtcvad
from groq import Groq
from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect

load_dotenv()

# ── Config ────────────────────────────────────────────────────
GROQ_API_KEY             = os.getenv('GROQ_API_KEY')
GOOGLE_STT_CREDENTIALS   = os.getenv('GOOGLE_STT_CREDENTIALS')
SAMPLE_RATE              = 16000
CHANNELS                 = 1
FRAME_DURATION_MS        = 30
SILENCE_THRESHOLD        = 27    # ~0.8 seconds

groq_client      = Groq(api_key=GROQ_API_KEY)
vad              = webrtcvad.Vad(2)
google_stt_client = None

# ── Google Cloud STT setup ────────────────────────────────────
if GOOGLE_STT_CREDENTIALS and os.path.exists(GOOGLE_STT_CREDENTIALS):
    try:
        from google.cloud import speech
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_STT_CREDENTIALS
        )
        google_stt_client = speech.SpeechClient(credentials=credentials)
        print("✅ Google Cloud STT enabled as PRIMARY for Bangla (bn-BD)")
    except Exception as e:
        print(f"⚠️  Google Cloud STT failed to initialise: {e}")
        print("    Falling back to Groq Whisper as primary")
else:
    print("ℹ️  No GOOGLE_STT_CREDENTIALS found — using Groq Whisper as primary")

print("✅ Groq Whisper ready as fallback STT")

# ── Known garbage patterns ────────────────────────────────────
WHISPER_HALLUCINATIONS = {
    "subtitles by the amara.org community",
    "amara.org",
    "www.moviewatcher.is",
    "transcript:",
    "transcribed by",
    "♪",
    "[music]",
    "[applause]",
    "[laughter]",
    "thank you for watching",
    "please subscribe",
}

# Plain strings (no r prefix) so \u escapes resolve to Unicode
WRONG_SCRIPT_PATTERNS = [
    '[\u0D80-\u0DFF]',   # Sinhala
    '[\u0600-\u06FF]',   # Arabic
    '[\u0400-\u04FF]',   # Cyrillic
    '[\uAC00-\uD7AF]',   # Korean
    '[\u4E00-\u9FFF]',   # Chinese
    '[\u3040-\u30FF]',   # Japanese
    '[\u0900-\u097F]',   # Devanagari (Hindi)
]

# ── Language detection ────────────────────────────────────────
def is_bangla(text: str) -> bool:
    bangla_chars = len(re.findall('[\u0980-\u09FF]', text))
    total_chars  = len(text.replace(' ', ''))
    if total_chars == 0:
        return False
    return (bangla_chars / total_chars) >= 0.3

def get_voice_token_limit(text: str) -> int:
    """
    Higher limits for voice calls – ensures responses are complete,
    especially for long Bangla explanations like the NGO program.
    Bangla needs ~3‑4 tokens per word vs ~1‑2 for English.
    """
    return 1024 if is_bangla(text) else 800

def fix_pronunciation(text: str) -> str:
    replacements = {
        "BelleVie": "Bell Vee",
        "bellevie": "Bell Vee",
        "BELLEVIE": "Bell Vee",
        "Sharmin":  "Sharr-meen",
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text

# ── Garbage transcript detection ──────────────────────────────
def is_garbage_transcript(text: str, expected_lang: str = None) -> bool:
    if not text or len(text.strip()) < 3:
        return True

    lower = text.strip().lower()

    for pattern in WHISPER_HALLUCINATIONS:
        if pattern in lower:
            print(f"🗑️ Hallucination detected: '{text[:60]}'")
            return True

    for script_pattern in WRONG_SCRIPT_PATTERNS:
        wrong_chars = len(re.findall(script_pattern, text))
        if wrong_chars > 2:
            print(f"🗑️ Wrong script detected: '{text[:60]}'")
            return True

    if expected_lang == "bn":
        bangla_ratio = len(re.findall('[\u0980-\u09FF]', text)) / max(len(text.replace(' ', '')), 1)
        if bangla_ratio < 0.15:
            print(f"🗑️ Expected Bangla but got ({bangla_ratio:.0%} Bangla): '{text[:60]}'")
            return True

    return False

# ── PCM to WAV ────────────────────────────────────────────────
def pcm_to_wav(pcm_data: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    return buf.getvalue()

# ── STT — Google Cloud (primary for Bangla) ───────────────────
async def transcribe_with_google(audio_bytes: bytes, lang_hint: str = None) -> str:
    if not google_stt_client:
        return ""
    try:
        from google.cloud import speech

        if lang_hint == "bn" or lang_hint is None:
            language_code      = "bn-BD"
            alt_language_codes = ["en-US"]
        else:
            language_code      = "en-US"
            alt_language_codes = ["bn-BD"]

        audio   = speech.RecognitionAudio(content=audio_bytes)
        config  = speech.RecognitionConfig(
            encoding                  = speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz         = SAMPLE_RATE,
            language_code             = language_code,
            alternative_language_codes= alt_language_codes,
            enable_automatic_punctuation = True,
            model                     = "latest_long",
        )

        def call_google():
            return google_stt_client.recognize(config=config, audio=audio)

        response   = await asyncio.to_thread(call_google)
        transcript = ""

        for result in response.results:
            transcript += result.alternatives[0].transcript + " "

        transcript = transcript.strip()
        print(f"📝 Google STT ({language_code}): {transcript}")
        return transcript

    except Exception as e:
        print(f"❌ Google STT error: {e}")
        return ""

# ── STT — Groq Whisper (fallback) ────────────────────────────
async def transcribe_with_groq(audio_bytes: bytes, lang_hint: str = None) -> str:
    try:
        wav_bytes = pcm_to_wav(audio_bytes)

        def call_whisper():
            params = {
                "file":            ("audio.wav", wav_bytes),
                "model":           "whisper-large-v3",
                "response_format": "text",
            }
            if lang_hint:
                params["language"] = lang_hint
            return groq_client.audio.transcriptions.create(**params)

        result     = await asyncio.to_thread(call_whisper)
        transcript = result.strip() if result else ""
        print(f"📝 Groq transcript ({lang_hint or 'auto'}): {transcript}")
        return transcript

    except Exception as e:
        print(f"❌ Groq transcription error: {e}")
        return ""

# ── Combined STT — Google primary, Groq fallback ──────────────
async def transcribe_audio(audio_bytes: bytes, lang_hint: str = None) -> str:
    if google_stt_client:
        transcript = await transcribe_with_google(audio_bytes, lang_hint)
        if transcript and not is_garbage_transcript(transcript, expected_lang=lang_hint):
            return transcript
        if transcript:
            print("⚠️ Google STT result looked unreliable — trying Groq...")
        else:
            print("⚠️ Google STT returned nothing — trying Groq...")

    return await transcribe_with_groq(audio_bytes, lang_hint)

# ── Text to speech ────────────────────────────────────────────
async def text_to_speech_audio(text: str) -> bytes:
    try:
        voice       = "bn-BD-NabanitaNeural" if is_bangla(text) else "en-US-JennyNeural"
        spoken_text = fix_pronunciation(text)
        print(f"🔊 TTS: {voice}")

        communicate = edge_tts.Communicate(spoken_text, voice, rate="+15%")
        buf = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])

        audio_bytes = buf.getvalue()
        print(f"🔊 TTS generated: {len(audio_bytes)} bytes")
        return audio_bytes

    except Exception as e:
        print(f"❌ TTS error: {e}")
        return b""

# ── VAD ───────────────────────────────────────────────────────
def is_speech(frame: bytes) -> bool:
    try:
        return vad.is_speech(frame, SAMPLE_RATE)
    except:
        return False

# ── Helper: wait for client audio_played ack ─────────────────
async def _wait_for_ack(ws, session_id):
    while True:
        data = await ws.receive()
        if data.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect(data.get("code", 1000))
        if "text" in data:
            text_data = data["text"]
            if text_data == "END_CALL":
                raise WebSocketDisconnect
            try:
                msg = json.loads(text_data)
                if msg.get("type") == "audio_played":
                    return
            except:
                pass

# ── Main voice handler ────────────────────────────────────────
async def handle_voice_call(
    websocket: WebSocket,
    session_id: str,
    chat_pipeline,
    sessions: dict
):
    await websocket.accept()
    print(f"📞 Call started: {session_id}")

    # ── Greeting ──────────────────────────────────────────────
    try:
        await websocket.send_json({
            "type":    "call_started",
            "message": "Connected to BelleVie AI Assistant"
        })

        greeting       = "Hello! Welcome to Bell Vee Global Health Services. I'm Sharmin, your AI health assistant. How can I help you today?"
        greeting_audio = await text_to_speech_audio(greeting)

        if greeting_audio:
            await websocket.send_json({"type": "mute"})
            await websocket.send_bytes(greeting_audio)
            await _wait_for_ack(websocket, session_id)
            await asyncio.sleep(0.3)
            await websocket.send_json({"type": "unmute"})
            await websocket.send_json({
                "type":    "listening",
                "message": "Listening..."
            })

    except WebSocketDisconnect:
        print(f"📵 Client disconnected during greeting: {session_id}")
        return
    except Exception as e:
        print(f"❌ Greeting error: {e}")
        return

    # ── Call state ────────────────────────────────────────────
    audio_buffer        = b""
    silence_count       = 0
    speech_detected     = False
    speech_buffer       = b""
    frame_size          = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2
    muted               = False
    pre_roll_history    = []
    max_pre_roll_frames = 8
    session_lang        = "bn"   # Bangla-first, switches to "en" if English detected

    try:
        while True:
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(data.get("code", 1000))

            if "text" in data:
                text_data = data["text"]
                if text_data == "END_CALL":
                    print(f"📵 Call ended by user: {session_id}")
                    goodbye       = "Thank you for calling Bell Vee. Take care and stay healthy! This is Sharmin, goodbye!"
                    goodbye_audio = await text_to_speech_audio(goodbye)
                    if goodbye_audio:
                        await websocket.send_json({"type": "mute"})
                        await websocket.send_bytes(goodbye_audio)
                        await _wait_for_ack(websocket, session_id)
                    break

                try:
                    msg = json.loads(text_data)
                    if msg.get("type") == "audio_played":
                        continue
                except:
                    pass
                continue

            chunk = data.get("bytes", b"")
            if not chunk or muted:
                continue

            audio_buffer += chunk

            while len(audio_buffer) >= frame_size:
                frame        = audio_buffer[:frame_size]
                audio_buffer = audio_buffer[frame_size:]

                speech_in_frame = is_speech(frame)

                if speech_in_frame:
                    if not speech_detected:
                        print(f"🎤 Speech started: {session_id}")
                        for f in pre_roll_history:
                            speech_buffer += f
                        pre_roll_history.clear()
                    speech_detected = True
                    silence_count   = 0
                    speech_buffer  += frame
                else:
                    if speech_detected:
                        silence_count += 1
                        speech_buffer += frame
                    else:
                        pre_roll_history.append(frame)
                        if len(pre_roll_history) > max_pre_roll_frames:
                            pre_roll_history.pop(0)

                if speech_detected and silence_count >= SILENCE_THRESHOLD:
                    print(f"⏸️ Silence detected — processing: {session_id}")

                    await websocket.send_json({"type": "mute"})
                    muted = True

                    await websocket.send_json({
                        "type":    "processing",
                        "message": "Processing..."
                    })

                    full_audio      = speech_buffer
                    speech_buffer   = b""
                    audio_buffer    = b""
                    speech_detected = False
                    silence_count   = 0
                    pre_roll_history.clear()

                    if len(full_audio) < SAMPLE_RATE // 2:
                        await websocket.send_json({
                            "type":    "listening",
                            "message": "Listening..."
                        })
                        await websocket.send_json({"type": "unmute"})
                        muted = False
                        continue

                    # ── STT ───────────────────────────────────
                    transcript = await transcribe_audio(full_audio, lang_hint=session_lang)

                    if not transcript.strip():
                        await websocket.send_json({
                            "type":    "listening",
                            "message": "Didn't catch that. Please speak again."
                        })
                        await websocket.send_json({"type": "unmute"})
                        muted = False
                        continue

                    # Update session language
                    if is_bangla(transcript):
                        if session_lang != "bn":
                            print("🌐 Language switched to: Bangla")
                        session_lang = "bn"
                    elif re.search(r'[a-zA-Z]', transcript) and not is_bangla(transcript):
                        if session_lang != "en":
                            print("🌐 Language switched to: English")
                        session_lang = "en"

                    # Garbage check
                    if is_garbage_transcript(transcript, expected_lang=session_lang):
                        await websocket.send_json({
                            "type":    "listening",
                            "message": "Didn't catch that clearly. Please speak again."
                        })
                        await websocket.send_json({"type": "unmute"})
                        muted = False
                        continue

                    await websocket.send_json({
                        "type": "transcript",
                        "text": transcript
                    })

                    # ── RAG ───────────────────────────────────
                    if session_id not in sessions:
                        sessions[session_id] = []

                    token_limit = get_voice_token_limit(transcript)

                    def run_chat_pipeline():
                        return chat_pipeline(
                            transcript,
                            sessions[session_id],
                            max_tokens=token_limit,
                            is_voice=True
                        )

                    response_text, sessions[session_id], model_used = await asyncio.to_thread(run_chat_pipeline)
                    print(f"🤖 [{model_used}] ({token_limit} tok) Response: {response_text[:80]}...")

                    await websocket.send_json({
                        "type": "response_text",
                        "text": response_text
                    })

                    # ── TTS ───────────────────────────────────
                    audio_response = await text_to_speech_audio(response_text)

                    if audio_response:
                        await websocket.send_bytes(audio_response)
                        await _wait_for_ack(websocket, session_id)
                        await asyncio.sleep(0.3)
                        await websocket.send_json({"type": "unmute"})
                        muted = False
                    else:
                        await websocket.send_json({
                            "type":    "error",
                            "message": "Could not generate audio. Please read the response above."
                        })
                        await websocket.send_json({"type": "unmute"})
                        muted = False

                    await websocket.send_json({
                        "type":    "listening",
                        "message": "Listening..."
                    })

    except WebSocketDisconnect:
        print(f"📵 Disconnected: {session_id}")
    except Exception as e:
        print(f"❌ Call error {session_id}: {e}")
        try:
            await websocket.send_json({
                "type":    "error",
                "message": "Call ended due to an error. Please try again."
            })
        except:
            pass
    finally:
        print(f"🔚 Call ended: {session_id}")