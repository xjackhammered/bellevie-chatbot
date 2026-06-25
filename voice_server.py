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
SILENCE_THRESHOLD = 50    # ~1.5 seconds of silence before processing

groq_client = Groq(api_key=GROQ_API_KEY)
vad         = webrtcvad.Vad(2)

# ── Language detection ────────────────────────────────────────
def is_bangla(text: str) -> bool:
    bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    total_chars  = len(text.replace(' ', ''))
    if total_chars == 0:
        return False
    return (bangla_chars / total_chars) >= 0.3

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
    try:
        wav_bytes = pcm_to_wav(audio_bytes)
        result    = groq_client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model="whisper-large-v3",
            response_format="text",
        )
        transcript = result.strip() if result else ""
        print(f"📝 Transcript: {transcript}")
        return transcript
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

# ── Text to speech ────────────────────────────────────────────
async def text_to_speech_audio(text: str) -> bytes:
    try:
        voice = "bn-BD-NabanitaNeural" if is_bangla(text) else "en-US-JennyNeural"
        print(f"🔊 TTS: {voice}")

        communicate = edge_tts.Communicate(text, voice)
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

# ── Helper: wait for client ack ───────────────────────────────
async def _wait_for_ack(ws, session_id):
    """Wait until client sends audio_played JSON message."""
    while True:
        data = await ws.receive()
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

    # ── Greeting (simple delay, no ack needed) ────────────
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
            await asyncio.sleep(2)        # let greeting play out
            await asyncio.sleep(0.3)      # guard period before unmuting
            await websocket.send_json({"type": "unmute"})
            await websocket.send_json({"type": "listening", "message": "Listening..."})

    except WebSocketDisconnect:
        print(f"📵 Client disconnected during greeting: {session_id}")
        return
    except Exception as e:
        print(f"❌ Greeting error: {e}")
        return

    # ── Call state ────────────────────────────────────────
    audio_buffer    = b""
    silence_count   = 0
    speech_detected = False
    speech_buffer   = b""
    frame_size      = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2
    muted           = False

    try:
        while True:
            data = await websocket.receive()

            # End call signal
            if "text" in data:
                text_data = data["text"]
                if text_data == "END_CALL":
                    print(f"📵 Call ended: {session_id}")
                    goodbye       = "Thank you for calling BelleVie. Take care and stay healthy!"
                    goodbye_audio = await text_to_speech_audio(goodbye)
                    if goodbye_audio:
                        await websocket.send_json({"type": "mute"})
                        await websocket.send_bytes(goodbye_audio)
                        await asyncio.sleep(2)
                    break
                # Ignore spurious audio_played in listening state
                try:
                    msg = json.loads(text_data)
                    if msg.get("type") == "audio_played":
                        continue
                except:
                    pass
                continue

            # Audio bytes — skip if muted
            chunk = data.get("bytes", b"")
            if not chunk or muted:
                continue

            audio_buffer += chunk

            # VAD processing
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

                # Silence threshold reached — user finished speaking
                if speech_detected and silence_count >= SILENCE_THRESHOLD:
                    print(f"⏸️ Processing speech: {session_id}")

                    # Immediately mute — stop accepting audio
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

                    # Skip if audio too short (under 0.5s)
                    if len(full_audio) < SAMPLE_RATE // 2:
                        await websocket.send_json({
                            "type":    "listening",
                            "message": "Listening..."
                        })
                        await websocket.send_json({"type": "unmute"})
                        muted = False
                        continue

                    # ── STT ───────────────────────────────
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

                    # ── RAG (voice uses max_tokens=400 for detailed responses) ──
                    if session_id not in sessions:
                        sessions[session_id] = []

                    # chat_pipeline returns (response, history, model_used)
                    response_text, sessions[session_id], model_used = chat_pipeline(
                        transcript,
                        sessions[session_id],
                        max_tokens=400
                    )
                    print(f"🤖 [{model_used}] Response: {response_text[:80]}...")

                    await websocket.send_json({
                        "type": "response_text",
                        "text": response_text
                    })

                    # ── TTS ───────────────────────────────
                    audio_response = await text_to_speech_audio(response_text)

                    if audio_response:
                        # Mic already muted — send audio
                        await websocket.send_bytes(audio_response)
                        # Wait for client to confirm playback finished
                        await _wait_for_ack(websocket, session_id)
                        # Short guard before reopening mic
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