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
SILENCE_THRESHOLD = 27    # 0.8 seconds — was 50 (1.5s), much more responsive

groq_client = Groq(api_key=GROQ_API_KEY)
vad         = webrtcvad.Vad(2)

# ── Language detection ────────────────────────────────────────
def is_bangla(text: str) -> bool:
    bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    total_chars  = len(text.replace(' ', ''))
    if total_chars == 0:
        return False
    return (bangla_chars / total_chars) >= 0.3

def detect_audio_language(audio_bytes: bytes) -> str | None:
    """
    Try to detect if the audio is likely Bangla or English
    by checking recent transcript history.
    Returns 'bn' for Bangla, None for auto-detect.
    This is a lightweight hint — Whisper still decides.
    """
    return None  # auto-detect by default — most accurate for mixed language

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
    Transcribe audio using Groq Whisper.
    No language hint — auto-detect is more accurate for
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
        result = await asyncio.to_thread(call_whisper)
        transcript = result.strip() if result else ""
        print(f" Transcript: {transcript}")
        return transcript
    except Exception as e:
        print(f" Transcription error: {e}")
        return ""

# ── Text to speech ────────────────────────────────────────────
async def text_to_speech_audio(text: str) -> bytes:
    """
    Convert text to MP3 using Edge TTS.
    rate='+15%' makes speech slightly faster — more natural for voice assistants.
    """
    try:
        voice = "bn-BD-NabanitaNeural" if is_bangla(text) else "en-US-JennyNeural"
        print(f" TTS: {voice}")

        communicate = edge_tts.Communicate(text, voice, rate="+15%")
        buf = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])

        audio_bytes = buf.getvalue()
        print(f" TTS generated: {len(audio_bytes)} bytes")
        return audio_bytes

    except Exception as e:
        print(f" TTS error: {e}")
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
    Wait until client sends audio_played JSON message.
    This guarantees the mic only reopens after the assistant
    has completely finished speaking — prevents echo.
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
        # ignore audio bytes while waiting for ack

# ── Main voice handler ────────────────────────────────────────
async def handle_voice_call(
    websocket: WebSocket,
    session_id: str,
    chat_pipeline,
    sessions: dict
):
    await websocket.accept()
    print(f" Call started: {session_id}")

    # ── Greeting ──────────────────────────────────────────────
    # IMPORTANT: mic stays muted the entire time the greeting plays.
    # We use _wait_for_ack so the mic ONLY opens after the greeting
    # audio has fully finished playing on the client.
    # This fixes the bug where "Listening..." appeared during greeting.
    try:
        await websocket.send_json({
            "type":    "call_started",
            "message": "Connected to BelleVie AI Assistant"
        })

        greeting       = "Hello! Welcome to BelleVie Global Health Services. I'm your AI health assistant. How can I help you today?"
        greeting_audio = await text_to_speech_audio(greeting)

        if greeting_audio:
            # Mute BEFORE sending greeting audio
            await websocket.send_json({"type": "mute"})
            await websocket.send_bytes(greeting_audio)

            # Wait for client to confirm greeting finished playing
            # Only THEN unmute — this fixes the "listening during greeting" bug
            await _wait_for_ack(websocket, session_id)

            # Small guard before opening mic — prevents capturing greeting tail
            await asyncio.sleep(0.3)
            await websocket.send_json({"type": "unmute"})
            await websocket.send_json({
                "type":    "listening",
                "message": "Listening..."
            })

    except WebSocketDisconnect:
        print(f" Client disconnected during greeting: {session_id}")
        return
    except Exception as e:
        print(f" Greeting error: {e}")
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
                    print(f" Call ended by user: {session_id}")
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
                        print(f" Speech started: {session_id}")
                    speech_detected = True
                    silence_count   = 0
                    speech_buffer  += frame
                else:
                    if speech_detected:
                        silence_count += 1
                        speech_buffer += frame

                # ── Silence threshold reached ─────────────────
                if speech_detected and silence_count >= SILENCE_THRESHOLD:
                    print(f" Processing speech: {session_id}")

                    # Mute immediately — stop accepting audio
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

                    # ── STT ───────────────────────────────────
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

                    # ── RAG ───────────────────────────────────
                    if session_id not in sessions:
                        sessions[session_id] = []

                    def run_chat_pipeline():
                        return chat_pipeline(
                            transcript,
                            sessions[session_id],
                            max_tokens=400
                        )
                    response_text, sessions[session_id], model_used = await asyncio.to_thread(run_chat_pipeline)
                    print(f" [{model_used}] Response: {response_text[:80]}...")

                    await websocket.send_json({
                        "type": "response_text",
                        "text": response_text
                    })

                    # ── TTS ───────────────────────────────────
                    audio_response = await text_to_speech_audio(response_text)

                    if audio_response:
                        # Mic already muted — send audio
                        await websocket.send_bytes(audio_response)

                        # Wait for client to confirm playback finished
                        await _wait_for_ack(websocket, session_id)

                        # Guard period before reopening mic
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
        print(f" Disconnected: {session_id}")
    except Exception as e:
        print(f" Call error {session_id}: {e}")
        try:
            await websocket.send_json({
                "type":    "error",
                "message": "Call ended due to an error. Please try again."
            })
        except:
            pass
    finally:
        print(f" Call ended: {session_id}")