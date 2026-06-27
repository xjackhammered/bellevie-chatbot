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
GROQ_API_KEY      = os.getenv('GROQ_API_KEY')
SAMPLE_RATE       = 16000
CHANNELS          = 1
FRAME_DURATION_MS = 30
SILENCE_THRESHOLD = 27    # 0.8 seconds — responsive but not too trigger-happy

groq_client = Groq(api_key=GROQ_API_KEY)
vad         = webrtcvad.Vad(2)

# ── Language detection ────────────────────────────────────────
def is_bangla(text: str) -> bool:
    bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    total_chars  = len(text.replace(' ', ''))
    if total_chars == 0:
        return False
    return (bangla_chars / total_chars) >= 0.3

def get_voice_token_limit(text: str) -> int:
    """
    Bangla script uses ~3-4 tokens per word vs ~1-2 for English.
    Give Bangla responses more room so they don't get cut off mid-sentence.
    """
    if is_bangla(text):
        return 600   # Bangla needs more tokens for the same amount of spoken content
    return 400       # English is fine at 400

# ── PCM to WAV ────────────────────────────────────────────────
def pcm_to_wav(pcm_data: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    return buf.getvalue()

# ── Speech to text ────────────────────────────────────────────
async def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe using Groq Whisper in a thread so it doesn't block
    the async event loop. Auto-detect language — most accurate for
    mixed English/Bangla/Banglish conversations.
    """
    try:
        wav_bytes = pcm_to_wav(audio_bytes)

        def call_whisper():
            return groq_client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model="whisper-large-v3",
                response_format="text",
            )

        result     = await asyncio.to_thread(call_whisper)
        transcript = result.strip() if result else ""
        print(f"📝 Transcript: {transcript}")
        return transcript

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

# ── Text to speech ────────────────────────────────────────────
async def text_to_speech_audio(text: str) -> bytes:
    """
    Convert text to MP3 using Edge TTS.
    rate='+15%' makes speech slightly faster — more natural for a voice assistant.
    """
    try:
        voice = "bn-BD-NabanitaNeural" if is_bangla(text) else "en-US-JennyNeural"
        print(f"🔊 TTS: {voice}")

        communicate = edge_tts.Communicate(text, voice, rate="+15%")
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
    """
    Wait until client sends audio_played JSON.
    Mic only reopens after assistant has completely finished speaking.
    """
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
        # ignore audio bytes while waiting

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
    # Mic stays muted until greeting fully finishes playing on client.
    # _wait_for_ack guarantees this — fixes "Listening during greeting" bug.
    try:
        await websocket.send_json({
            "type":    "call_started",
            "message": "Connected to BelleVie AI Assistant"
        })

        greeting       = "Hello! Welcome to BelleVie Global Health Services. I'm your AI health assistant. How can I help you today?"
        greeting_audio = await text_to_speech_audio(greeting)

        if greeting_audio:
            await websocket.send_json({"type": "mute"})
            await websocket.send_bytes(greeting_audio)
            await _wait_for_ack(websocket, session_id)   # mic opens only after greeting finishes
            await asyncio.sleep(0.3)                      # guard period
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
    audio_buffer    = b""
    silence_count   = 0
    speech_detected = False
    speech_buffer   = b""
    frame_size      = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2
    muted           = False

    try:
        while True:
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(data.get("code", 1000))

            # End call signal
            if "text" in data:
                text_data = data["text"]
                if text_data == "END_CALL":
                    print(f"📵 Call ended by user: {session_id}")
                    goodbye       = "Thank you for calling BelleVie. Take care and stay healthy!"
                    goodbye_audio = await text_to_speech_audio(goodbye)
                    if goodbye_audio:
                        await websocket.send_json({"type": "mute"})
                        await websocket.send_bytes(goodbye_audio)
                        await _wait_for_ack(websocket, session_id)
                    break

                # Ignore spurious audio_played in listening state
                try:
                    msg = json.loads(text_data)
                    if msg.get("type") == "audio_played":
                        continue
                except:
                    pass
                continue

            # Audio bytes — skip completely if muted
            chunk = data.get("bytes", b"")
            if not chunk or muted:
                continue

            audio_buffer += chunk

            # ── VAD processing ────────────────────────────────
            while len(audio_buffer) >= frame_size:
                frame        = audio_buffer[:frame_size]
                audio_buffer = audio_buffer[frame_size:]

                speech_in_frame = is_speech(frame)

                if speech_in_frame:
                    if not speech_detected:
                        print(f"🎤 Speech started: {session_id}")
                    speech_detected = True
                    silence_count   = 0
                    speech_buffer  += frame
                else:
                    if speech_detected:
                        silence_count += 1
                        speech_buffer += frame

                # ── Silence threshold reached ─────────────────
                if speech_detected and silence_count >= SILENCE_THRESHOLD:
                    print(f"⏸️ Processing speech: {session_id}")

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

                    # Skip if audio too short (under 0.5 seconds)
                    if len(full_audio) < SAMPLE_RATE // 2:
                        await websocket.send_json({
                            "type":    "listening",
                            "message": "Listening..."
                        })
                        await websocket.send_json({"type": "unmute"})
                        muted = False
                        continue

                    # ── STT (non-blocking) ────────────────────
                    transcript = await transcribe_audio(full_audio)

                    if not transcript.strip():
                        await websocket.send_json({
                            "type":    "listening",
                            "message": "Didn't catch that. Please speak again."
                        })
                        await websocket.send_json({"type": "unmute"})
                        muted = False
                        continue

                    await websocket.send_json({
                        "type": "transcript",
                        "text": transcript
                    })

                    # ── RAG (non-blocking, token limit based on language) ──
                    if session_id not in sessions:
                        sessions[session_id] = []

                    # Detect language from transcript to set correct token limit
                    # Bangla gets 600 tokens, English gets 400
                    token_limit = get_voice_token_limit(transcript)

                    def run_chat_pipeline():
                        return chat_pipeline(
                            transcript,
                            sessions[session_id],
                            max_tokens=token_limit
                        )

                    response_text, sessions[session_id], model_used = await asyncio.to_thread(run_chat_pipeline)
                    print(f"🤖 [{model_used}] ({token_limit} tokens) Response: {response_text[:80]}...")

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