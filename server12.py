from pipeline import AudioPipeline
import json, base64
import asyncio, json, base64, audioop, subprocess, os, re, time
from urllib.parse import parse_qs, urlparse
from collections import deque
from typing import List, Optional, Dict, Any, Optional, TYPE_CHECKING, TypeVar
import logging as log

# For type hints
#TTSController = TypeVar('TTSController') 

import numpy as np, torch, whisper, boto3
import httpx
import websockets
from websockets.asyncio.server import serve, ServerConnection as WebSocket

from noise import denoise_pcm16
from session_logger import Session
from config import AWS_REGION, POLLY_VOICE

# ------------------- ElevenLabs Integration -------------------
from elevenlabs import ElevenLabs

# ------------------- VKG / RAG: Neo4j Semantic Search (KEPT, but only used for rag_query) -------------------
from knowledge_retrieve import semantic_search, VECTOR_INDEX_NAME

ELEVEN_API_KEY = "sk_fbf4c83e0e5788a6e705136f996e388a7f11269acdae7b0c"
ELEVEN_MODEL_ID = "eleven_multilingual_v2"
ELEVEN_LATENCY = 3

ELEVEN_VOICES = {
    "en": "dMyQqiVXTU80dDl2eNK8",
    "it": "CiwzbDpaN3pQXjTgx3ML",
    "fr": "F1toM6PcP54s45kOOAyV",
    "de": "v3V1d2rk6528UrLKRuy8",
}
_eleven = ElevenLabs(api_key=ELEVEN_API_KEY)

LLAMA_MODEL = os.getenv("LLAMA_MODEL", "clinicbot-llama3")
LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "8765"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))  # sensible default
RAG_INCLUDE_ASSIST = os.getenv("RAG_INCLUDE_ASSIST", "false").lower() in ("1", "true", "yes")
FRAME_BYTES = 256 * 2
VAD_SR = 8000
WHISPER_SR = 16000
INACTIVITY_TIMEOUT = 10 # seconds

RESP_MAX_WORDS = int(os.getenv("RESP_MAX_WORDS", "35"))  # Reduced for brevity

# --- CRITICAL CHANGE FOR MULTILINGUAL ACCURACY ---
# Changed WHISPER_MODEL from "small" to "medium".
# The "medium" model is larger but significantly more accurate at detecting the correct
# language from audio, which is the key to providing responses in the correct language.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")

RAG_INDEX_NAME = os.getenv("RAG_INDEX_NAME", VECTOR_INDEX_NAME or "vector_entities_index")

# ---- RAG Config (VKG) ----
RAG_INDEX_NAME = os.getenv("RAG_INDEX_NAME", VECTOR_INDEX_NAME or "vector_entities_index")
RAG_DOC_ID = os.getenv("RAG_DOC_ID", "SwissBot")
FAQ_DOC_ID = os.getenv("FAQ_DOC_ID", "ClinicFAQ")
MEDICAL_ENCYC_DOC_ID = os.getenv("MEDICAL_ENCYC_DOC_ID", "ClinicMedKB")  # <— NEW: general medical knowledge base


# ---- LLM Intent Classifier Config ----
INTENT_LABELS = [
    "general",              # 0 (fallback)
    "medical_advice",       # 1  <- NEW
    "medical_question",     # 2  <- NEW
    "critical",             # 3
    "suicidal",             # 4
    "violence",             # 5
    "rude",                 # 6
    "chitchat",             # 7
    "rag_query",            # 8 (clinic info via VKG)
    "appointment_booking",  # 9
    "callback_request",     # 10 <- NEW for callback requests
    "speak_to_someone",     # 11 <- NEW: wants to talk to staff/doctor now
    "prescription_renewal", # 12 <- NEW: renew/refill prescriptions
    "appointment_reschedule", # 13 <- NEW: reschedule an existing appointment
    "appointment_cancellation" # 14 <- NEW: cancel an existing appointment
]
LABEL2ID = {lab: i for i, lab in enumerate(INTENT_LABELS)}


INTENT_MAXLEN = int(os.getenv("INTENT_MAXLEN", "256"))

# --------- SLOT-FILLING ORCHESTRATOR PROMPT ----------
SYSTEM_PROMPT = """
You are a calm, efficient medical receptionist.
Principles:
- Understand what the caller already said. NEVER ask for info they already provided.
- Fill missing details ONE at a time with a short, natural question.
- When enough info is gathered, give a brief confirmation (one or two sentences), then proceed.
- Be concise. Avoid filler. No lists or numbered steps.
- When mentioning times or small numbers, write them in words.
- If the user asks broad medical info, stay general and invite booking; no diagnosis or dosing.
- Safety first: emergencies → tell them to call the emergency line.
- MAXIMUM response length is {RESP_MAX_WORDS} words. SHORTER IS BETTER.
- Respond ONLY in the user's language.
"""

def SYS():
    return SYSTEM_PROMPT.format(RESP_MAX_WORDS=RESP_MAX_WORDS)

BAD_WORDS = {"en":{"fuck","shit"},"fr":{"merde","putain"},"de":{"scheiße","fick"},"it":{"cazzo","merda"}}

GOODBYE = re.compile(r"\b(bye|good\s*bye|bye\s*now|see\s*ya|hang\s*up|i'?ll\s*(?:go|leave)|i\s*(?:will|wanna)\s*(?:go|leave)|au\s*revoir|tsch[üu]ss|auf\s*wiedersehen|arrivederci|ciao)\b", re.I)
LANG_TAG_RX = re.compile(r'</?lang[^>]*>', re.I)
LEADING_NUM_RX = re.compile(r'^\s*\d+[.]\s*')
VOICE_OVERRIDES = {"de": os.getenv("DE_VOICE", "Daniel")}

polly = boto3.client("polly", region_name=AWS_REGION)

# Load VAD + Whisper
# vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
# wh = whisper.load_model(WHISPER_MODEL)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import redis
REDIS_URI = os.getenv("REDIS_URI", "redis://127.0.0.1:6379/0")
STREAM_KEY = os.getenv("REDIS_STREAM", "clinicbot:events")
try: 
    _r = redis.Redis.from_url(REDIS_URI, decode_responses=True)
    # Test connection
    _r.ping()
except Exception: 
    _r = None
    print("Warning: Redis connection failed. Callback storage will be disabled.")

def emit(kind: str, **fields):
    if not _r: return
    data = {"kind": kind, "ts": str(int(time.time() * 1000))}
    for k, v in fields.items():
        data[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else ("" if v is None else str(v))
    try: _r.xadd(STREAM_KEY, data, maxlen=10000, approximate=True)
    except Exception: pass

# ---- Twilio helpers: resolve caller phone number ----
def _normalize_phone_number(raw: Optional[str]) -> str:
    if not isinstance(raw, str):
        return ""
    cleaned = re.sub(r"[^\d+]", "", raw)
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]
    return cleaned

def extract_phone_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    try:
        m = re.search(r"[\d\s\-\(\)\.+]{7,}", text)
        if not m:
            return None
        return _normalize_phone_number(m.group())
    except Exception:
        return None

def extract_date_time_from_text(user_text: str, lang: str) -> tuple[Optional[str], Optional[str]]:
    """LLM-assisted extraction of calendar date and 24h time from a short user message."""
    try:
        prompt = (
            f"Extract a calendar date and a 24-hour time from the user's last message.\\n"
            f"Return STRICT JSON: {{\"date_iso\": \"YYYY-MM-DD or null\", \"time_24h\": \"HH:MM or null\"}}. No extra text.\\n\\n"
            f"USER: {user_text}\\n"
            f"JSON:"
        )
        raw = llama(prompt, lang)
        data = _extract_json(raw) or {}
        date_iso = data.get("date_iso") if isinstance(data.get("date_iso"), str) else None
        time_24h = data.get("time_24h") if isinstance(data.get("time_24h"), str) else None
        return date_iso, time_24h
    except Exception:
        return None, None

async def resolve_twilio_caller_number(start_obj: Dict[str, Any]) -> str:
    """Best-effort extraction of caller phone from Twilio Voice Media Stream start event.

    Priority:
    1) start.customParameters.From (if you passed <Parameter name="From" value="{{From}}"/>)
    2) Twilio REST lookup via callSid
    3) empty string on failure
    """
    try:
        custom = start_obj.get("customParameters") or {}
        if isinstance(custom, dict):
            for key in ("From", "from", "caller", "phone", "number"):
                val = custom.get(key)
                if val:
                    return _normalize_phone_number(val)

        call_sid = start_obj.get("callSid") or start_obj.get("call_sid")
        if call_sid:
            acc = os.getenv("TWILIO_ACCOUNT_SID")
            tok = os.getenv("TWILIO_AUTH_TOKEN")
            if acc and tok:
                url = f"https://api.twilio.com/2010-04-01/Accounts/{acc}/Calls/{call_sid}.json"
                try:
                    async with httpx.AsyncClient(timeout=6.0) as client:
                        resp = await client.get(url, auth=(acc, tok))
                    if resp.status_code == 200:
                        data = resp.json()
                        frm = data.get("from") or data.get("from_formatted") or data.get("caller")
                        return _normalize_phone_number(frm)
                except Exception as e:
                    print(f"Twilio lookup failed for {call_sid}: {e}")
    except Exception:
        pass
    return ""

def generate_conversation_summary(conversation_history: str, lang: str) -> str:
    """Generate a summary of the conversation for callback purposes."""
    summary_prompt = f"""
    Summarize this conversation between a patient and a clinic receptionist bot in {lang}.
    Focus on the main concerns, questions asked, and any relevant medical or appointment-related topics.
    Keep it concise and professional, around 2-3 sentences.
    
    Conversation:
    {conversation_history}
    
    Summary:
    """
    
    try:
        return clean_reply(llama(summary_prompt, lang)).strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Conversation summary unavailable due to processing error."

def store_callback_request(stream_sid: str, phone_number: str, conversation_history: str, summary: str, lang: str) -> bool:
    """Store callback request information in Redis."""
    if not _r:
        print("Redis not available. Cannot store callback request.")
        return False
    
    try:
        callback_data = {
            "status": "request_callback",
            "phone_number": phone_number,
            "conversation_history": conversation_history,
            "summary": summary,
            "description": summary,
            "language": lang,
            "timestamp": str(int(time.time())),
            "processed": "false"
        }
        
        # Store with SID as the ONLY key (no prefixes)
        _r.hset(stream_sid, mapping=callback_data)
        print(f"Callback request stored for SID: {stream_sid}")
        return True
    except Exception as e:
        print(f"Error storing callback request: {e}")
        return False

class TurnManager:
    def __init__(self, max_pairs: int = 6):
        self.user_turn = True
        self.events = deque(maxlen=max_pairs)
        self.lang: Optional[str] = None
        self.awaiting_doctor = False
        self.callback_state = None  # None, 'offered', 'waiting_response'
        # Track whether appointment booking intent occurred during session
        self.appointment_detected = False
        # Track prescription renewal flow state: None | 'waiting_name' | 'completed'
        self.prescription_state = None
        # Generic phone request flag to avoid repeating question
        self.phone_pending = False
        # Slot-filling orchestrator state
        self.slots: Dict[str, Any] = {}
        self.intent: Optional[str] = None

    def allow_user_turn(self) -> bool: return self.user_turn
    def start_bot_turn(self): self.user_turn = False
    def finish_bot_turn(self): self.user_turn = True

    def log_pair(self, user_text: str, bot_text: str, class_id: Optional[int] = None, label: Optional[str] = None):
        self.events.append({"user": user_text, "bot": bot_text, "class_id": class_id, "label": label})

    def build_context_text(self, max_words: int = None) -> str:
        print("[Building Context Text - Full History]")
        ctx = deque()
        for e in reversed(self.events):
            pair = f"User: {e['user']}\nBot: {e['bot']}"
            ctx.appendleft(pair)
        return "\n".join(ctx)
    
    # ----------------- TTS Controller -----------------
class TTSController:
    """Manages non-blocking, interruptible TTS streaming."""
    def __init__(self):
        self.current_generator = None
        self.is_speaking = False
        self.audio_queue: Optional[asyncio.Queue] = None
        self.producer_task: Optional[asyncio.Task] = None

    async def stop_immediately(self):
        """Force-stops any ongoing TTS generation and sending."""
        if not self.is_speaking and not self.producer_task:
            return

        log.debug("[TTS] Attempting to stop TTS immediately.")
        self.is_speaking = False

        if self.producer_task and not self.producer_task.done():
            self.producer_task.cancel()
            try:
                await self.producer_task
            except asyncio.CancelledError:
                log.debug("[TTS] Producer task successfully cancelled.")
        
        if self.current_generator:
            try: 
                self.current_generator.close()
            except Exception: 
                pass
        
        self.current_generator = None
        self.producer_task = None
        
        if self.audio_queue:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
            log.debug("[TTS] Audio queue has been cleared.")

tts_controller = TTSController() 

EXPLICIT_SWITCH = {
    "de": [r"\bdeutsch\b",r"\bsprechen sie deutsch\b",r"\bauf deutsch\b"],
    "fr": [r"\bfran[cç]ais\b",r"\ben fran[cç]ais\b"],
    "en": [r"\benglish\b",r"\bin english\b"],
    "it": [r"\bitaliano\b",r"\bparli italiano\b",r"\bin italiano\b"]
}

def detect_explicit_switch(text:str):
    for code,pats in EXPLICIT_SWITCH.items():
        low=text.lower()
        for p in pats:
            if re.search(p,low): return code
    return None

def choose_lang(sess_lang, whisper_lang, text):
    sw=detect_explicit_switch(text)
    if sw: return sw
    return (whisper_lang or sess_lang or "fr")[:2]

def clean_reply(txt:str)->str:
    txt = LANG_TAG_RX.sub("", txt); return LEADING_NUM_RX.sub("", txt).strip()

def truncate_words(text:str, max_words:int=RESP_MAX_WORDS)->str:
    words=text.split()
    return text if len(words)<=max_words else " ".join(words[:max_words]).rstrip(",.;:! ")+ "..."

def llama(prompt:str, lang:str, timeout:float=45.0)->str:
    tagged=f"<lang {lang}>\n{prompt}"; base=["ollama","run",LLAMA_MODEL]
    print(f"[LLAMA] Running command: {' '.join(base)} with timeout {timeout}s")
    def _run(cmd):
        p=subprocess.run(cmd,input=tagged.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=True)
        print("LLAMA raw output:", p.stdout.decode().strip())
        return p.stdout.decode().strip()
    try: return _run(base+["--keep-alive","0"])
    except subprocess.CalledProcessError as e:
        if b"unknown flag" in e.stderr or b"flag provided" in e.stderr: return _run(base)
        raise
    except subprocess.TimeoutExpired:
        return {"en":"Sorry, I had a delay. Could you repeat that?","fr":"Désolé, j'ai eu un délai. Pouvez-vous répéter ?","de":"Entschuldigung, es gab eine Verzögerung. Können Sie das wiederholen?","it":"Scusa, ho avuto un ritardo. Puoi ripetere?"}.get(lang,"Sorry, please repeat that?")

def caller(ws)->str:
    hdrs=getattr(ws,"request_headers",{}) or getattr(ws,"headers",{}); num=hdrs.get("From")
    if num: return num
    q=urlparse(getattr(ws,"path","")).query
    return parse_qs(q).get("From",["?"])[0]

# ------------------- SLOT DETECTION FUNCTIONS -------------------

def detect_appointment_type(text: str, history: str = "") -> Optional[tuple[str, int]]:
    """Detect appointment type and return (type, duration_minutes)."""
    combined = f"{history} {text}".lower()
    
    # Map variations to canonical types with durations
    type_patterns = {
        "first_consultation": (r"\b(first\s+(consultation|visit|appointment)|new\s+patient|initial\s+(consultation|visit))\b", 40),
        "follow_up": (r"\b(follow\s*up|control\s+(visit|appointment)|review|check\s*up|revisit)\b", 20),
        "iron_infusion": (r"\b(iron\s*(infusion|fusion|infus)|infusion\s*iron)\b", 20),
        "pragmafare": (r"\bpragmafare\b", 30),
        "other_scheduling": (r"\b(other|general|regular|standard|normal)\s*(scheduling|appointment)\b", 20)
    }
    
    for apt_type, (pattern, duration) in type_patterns.items():
        if re.search(pattern, combined):
            return apt_type, duration
    
    return None

def detect_time_and_window(text: str) -> Dict[str, Any]:
    """Detect time preferences, weekdays, and relative windows."""
    result = {}
    text_lower = text.lower()
    
    # Specific time patterns (24h format)
    time_patterns = [
        r"\b(\d{1,2}):(\d{2})\b",
        r"\b(\d{1,2})\s*(am|pm)\b",
        r"\b(eight|nine|ten|eleven|twelve|one|two|three|four|five|six|seven)\s*(am|pm|o'?clock)\b"
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Simple conversion for common times
            time_words = {
                "eight": "08:00", "nine": "09:00", "ten": "10:00", "eleven": "11:00",
                "twelve": "12:00", "one": "13:00", "two": "14:00", "three": "15:00",
                "four": "16:00", "five": "17:00", "six": "18:00", "seven": "19:00"
            }
            if match.group(1) in time_words:
                result["time_24h"] = time_words[match.group(1)]
            break
    
    # Time preferences
    if re.search(r"\bmorning\b", text_lower):
        result["time_pref"] = "morning"
    elif re.search(r"\bafternoon\b", text_lower):
        result["time_pref"] = "afternoon"
    elif re.search(r"\bevening\b", text_lower):
        result["time_pref"] = "evening"
    
    # Weekdays
    weekday_patterns = {
        "monday": r"\bmonday\b", "tuesday": r"\btuesday\b", "wednesday": r"\bwednesday\b",
        "thursday": r"\bthursday\b", "friday": r"\bfriday\b", "saturday": r"\bsaturday\b",
        "sunday": r"\bsunday\b"
    }
    
    for day, pattern in weekday_patterns.items():
        if re.search(pattern, text_lower):
            result["day_of_week"] = day.capitalize()
            break
    
    # Relative windows
    if re.search(r"\bnext\s+week\b", text_lower):
        result["next_week_flag"] = True
    elif re.search(r"\bthis\s+week\b", text_lower):
        result["this_week_flag"] = True
    elif re.search(r"\btoday\b", text_lower):
        result["today_flag"] = True
    elif re.search(r"\btomorrow\b", text_lower):
        result["tomorrow_flag"] = True
    
    return result

def detect_date_iso(text: str, lang: str = "en") -> Optional[str]:
    """Detect and convert dates to ISO format."""
    from datetime import datetime, timedelta
    
    text_lower = text.lower()
    now = datetime.now()
    
    # Direct date patterns (DD/MM/YYYY, DD-MM-YYYY, etc.)
    date_patterns = [
        r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
        r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b"
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                if len(match.group(1)) == 4:  # YYYY-MM-DD format
                    return f"{match.group(1)}-{match.group(2):0>2}-{match.group(3):0>2}"
                else:  # DD/MM/YYYY format
                    return f"{match.group(3)}-{match.group(2):0>2}-{match.group(1):0>2}"
            except:
                pass
    
    # Weekday conversion using existing logic
    time_window = detect_time_and_window(text)
    if time_window.get("day_of_week"):
        weekday_name = time_window["day_of_week"].lower()
        weekday_idx = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].index(weekday_name)
        
        today_idx = now.weekday()
        delta = (weekday_idx - today_idx) % 7
        
        if delta == 0 and not time_window.get("today_flag"):
            delta = 7  # Next week if same day but not explicitly "today"
        
        target_date = now.date() + timedelta(days=delta)
        return target_date.strftime("%Y-%m-%d")
    
    return None

def detect_patient_status(text: str, history: str = "") -> Optional[str]:
    """Detect if patient is new or existing."""
    combined = f"{history} {text}".lower()
    
    if re.search(r"\b(first\s+(consultation|visit|appointment)|new\s+patient|never\s+been|first\s+time)\b", combined):
        return "new"
    elif re.search(r"\b(follow\s*up|return|existing|regular|been\s+before|last\s+time)\b", combined):
        return "existing"
    
    return None

def detect_name(text: str, history: str = "") -> Optional[str]:
    """Extract patient name from text or history."""
    # Try the existing LLM extraction first
    extracted = _extract_name_from_text(text, history, "en")
    if extracted:
        return extracted
    
    # Simple regex fallback for common patterns
    name_patterns = [
        r"\bi'?m\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b",
        r"\bmy\s+name\s+is\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b",
        r"\bthis\s+is\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).title()
    
    return None

def detect_phone(text: str) -> Optional[str]:
    """Extract phone number from text."""
    return extract_phone_from_text(text)

def detect_previous_appointment(text: str) -> Dict[str, Any]:
    """Detect references to existing appointments for reschedule/cancel."""
    result = {}
    
    # Look for date references
    prev_date = detect_date_iso(text)
    if prev_date:
        result["prev_date_iso"] = prev_date
    
    # Look for time references  
    time_info = detect_time_and_window(text)
    if time_info.get("time_24h"):
        result["prev_time_24h"] = time_info["time_24h"]
    
    # Look for "the only" or "my" appointment
    if re.search(r"\b(the\s+only|my\s+only|my\s+appointment)\b", text.lower()):
        result["single_appointment"] = True
    
    return result

# ------------------- SLOT SCHEMAS AND ORCHESTRATOR -------------------

SLOT_SCHEMAS = {
    "appointment_booking": [
        "appointment_type", "date_iso", "time_24h", "time_pref", 
        "patient_status", "patient_name", "phone_number"
    ],
    "appointment_reschedule": [
        "prev_date_iso", "prev_time_24h", "new_date_iso", "new_time_24h", 
        "time_pref", "appointment_type", "patient_name", "phone_number"
    ],
    "appointment_cancellation": [
        "prev_date_iso", "prev_time_24h", "patient_name", "phone_number"
    ],
    "prescription_renewal": [
        "patient_name", "phone_number"
    ],
    "callback_request": [
        "phone_number"
    ]
}

SLOT_PRIORITY = {
    "appointment_booking": ["appointment_type", "date_iso", "time_pref", "patient_status", "patient_name", "phone_number"],
    "appointment_reschedule": ["prev_date_iso", "new_date_iso", "time_pref", "patient_name", "phone_number"],
    "appointment_cancellation": ["prev_date_iso", "patient_name", "phone_number"],
    "prescription_renewal": ["patient_name", "phone_number"],
    "callback_request": ["phone_number"]
}

def extract_all_slots(user_text: str, history: str, lang: str) -> Dict[str, Any]:
    """Extract all possible slots from current text and history."""
    slots = {}
    
    # Appointment type
    apt_type_result = detect_appointment_type(user_text, history)
    if apt_type_result:
        slots["appointment_type"] = apt_type_result[0]
        slots["duration_minutes"] = apt_type_result[1]
    
    # Time and date information
    time_info = detect_time_and_window(user_text)
    if time_info.get("time_24h"):
        slots["time_24h"] = time_info["time_24h"]
    if time_info.get("time_pref"):
        slots["time_pref"] = time_info["time_pref"]
    if time_info.get("day_of_week"):
        slots["day_of_week"] = time_info["day_of_week"]
    
    # Date detection
    date_iso = detect_date_iso(user_text, lang)
    if date_iso:
        slots["date_iso"] = date_iso
        # Also try to set new_date_iso for reschedule scenarios
        slots["new_date_iso"] = date_iso
    
    # Patient status
    patient_status = detect_patient_status(user_text, history)
    if patient_status:
        slots["patient_status"] = patient_status
    
    # Name and phone
    name = detect_name(user_text, history)
    if name:
        slots["patient_name"] = name
    
    phone = detect_phone(user_text)
    if phone:
        slots["phone_number"] = phone
    
    # Previous appointment info for reschedule/cancel
    prev_apt = detect_previous_appointment(user_text)
    if prev_apt:
        slots.update(prev_apt)
    
    return slots

def get_missing_slots(intent: str, current_slots: Dict[str, Any]) -> List[str]:
    """Get list of missing required slots for the intent."""
    required = SLOT_SCHEMAS.get(intent, [])
    missing = []
    
    for slot in required:
        # Special logic for alternative slots
        if slot == "date_iso" and current_slots.get("day_of_week"):
            continue  # day_of_week can substitute for date_iso
        elif slot == "time_24h" and current_slots.get("time_pref"):
            continue  # time_pref can substitute for specific time
        elif slot == "new_date_iso" and current_slots.get("date_iso"):
            continue  # date_iso can be used as new_date_iso
        elif slot == "prev_date_iso" and current_slots.get("single_appointment"):
            continue  # single appointment doesn't need specific date
        elif not current_slots.get(slot):
            missing.append(slot)
    
    return missing

def generate_slot_question(missing_slot: str, lang: str, current_slots: Dict[str, Any]) -> str:
    """Generate a natural question for the missing slot."""
    questions = {
        "appointment_type": {
            "en": "What type of appointment do you need?",
            "fr": "Quel type de rendez-vous souhaitez-vous ?",
            "de": "Welche Art von Termin benötigen Sie?",
            "it": "Che tipo di appuntamento ti serve?"
        },
        "date_iso": {
            "en": "Which day works best for you?",
            "fr": "Quel jour vous convient le mieux ?",
            "de": "Welcher Tag passt Ihnen am besten?",
            "it": "Quale giorno va meglio per te?"
        },
        "time_pref": {
            "en": "Do you prefer morning, afternoon, or evening?",
            "fr": "Préférez-vous le matin, l'après-midi ou le soir ?",
            "de": "Bevorzugen Sie morgens, nachmittags oder abends?",
            "it": "Preferisci mattina, pomeriggio o sera?"
        },
        "patient_name": {
            "en": "What's your name?",
            "fr": "Quel est votre nom ?",
            "de": "Wie ist Ihr Name?",
            "it": "Come ti chiami?"
        },
        "phone_number": {
            "en": "What's your phone number?",
            "fr": "Quel est votre numéro de téléphone ?",
            "de": "Wie ist Ihre Telefonnummer?",
            "it": "Qual è il tuo numero di telefono?"
        },
        "prev_date_iso": {
            "en": "Which appointment would you like to change?",
            "fr": "Quel rendez-vous souhaitez-vous modifier ?",
            "de": "Welchen Termin möchten Sie ändern?",
            "it": "Quale appuntamento vuoi cambiare?"
        },
        "new_date_iso": {
            "en": "What's your preferred new date?",
            "fr": "Quelle nouvelle date préférez-vous ?",
            "de": "Welches neue Datum bevorzugen Sie?",
            "it": "Quale nuova data preferisci?"
        }
    }
    
    lang_key = lang[:2] if lang else "en"
    return questions.get(missing_slot, {}).get(lang_key, questions[missing_slot]["en"])

def generate_confirmation(intent: str, slots: Dict[str, Any], lang: str) -> str:
    """Generate confirmation message when all slots are filled."""
    confirmations = {
        "appointment_booking": {
            "en": "Perfect! I'll book your {type} for {day}. Your name and phone number?",
            "fr": "Parfait ! Je réserve votre {type} pour {day}. Votre nom et numéro ?",
            "de": "Perfekt! Ich buche Ihren {type} für {day}. Ihr Name und Telefonnummer?",
            "it": "Perfetto! Prenoto il tuo {type} per {day}. Nome e numero di telefono?"
        },
        "appointment_reschedule": {
            "en": "I'll reschedule your appointment to {day}. Confirming with your details.",
            "fr": "Je reporte votre rendez-vous à {day}. Confirmation avec vos coordonnées.",
            "de": "Ich verschiebe Ihren Termin auf {day}. Bestätigung mit Ihren Daten.",
            "it": "Sposto il tuo appuntamento a {day}. Confermo con i tuoi dati."
        },
        "appointment_cancellation": {
            "en": "I'll cancel your appointment. Just need to confirm your details.",
            "fr": "J'annule votre rendez-vous. Juste besoin de confirmer vos coordonnées.",
            "de": "Ich storniere Ihren Termin. Ich muss nur Ihre Daten bestätigen.",
            "it": "Annullo il tuo appuntamento. Devo solo confermare i tuoi dati."
        },
        "prescription_renewal": {
            "en": "I'll request your prescription renewal. The doctor will contact you soon.",
            "fr": "Je demande le renouvellement de votre ordonnance. Le médecin vous contactera bientôt.",
            "de": "Ich beantrage die Erneuerung Ihres Rezepts. Der Arzt wird Sie bald kontaktieren.",
            "it": "Richiedo il rinnovo della tua prescrizione. Il medico ti contatterà presto."
        },
        "callback_request": {
            "en": "I'll arrange for someone to call you back as soon as possible.",
            "fr": "Je vais organiser pour que quelqu'un vous rappelle dès que possible.",
            "de": "Ich werde dafür sorgen, dass Sie so schnell wie möglich zurückgerufen werden.",
            "it": "Organizzerò perché qualcuno ti richiami il prima possibile."
        }
    }
    
    lang_key = lang[:2] if lang else "en"
    template = confirmations.get(intent, {}).get(lang_key, confirmations[intent]["en"])
    
    # Simple template substitution
    day_info = slots.get("day_of_week", "").lower()
    apt_type = slots.get("appointment_type", "appointment").replace("_", " ")
    
    return template.format(type=apt_type, day=day_info)

def build_orchestrated_reply(label: str, user_text: str, history: str, lang: str, sess) -> str:
    """Main orchestrator function for slot-filling conversations."""
    
    # Set intent if not already set
    if not sess.turns.intent:
        sess.turns.intent = label
    
    # Extract new slots from current turn
    new_slots = extract_all_slots(user_text, history, lang)
    
    # Merge with existing slots (don't overwrite non-empty values)
    for key, value in new_slots.items():
        if value and not sess.turns.slots.get(key):
            sess.turns.slots[key] = value
    
    # Get missing slots
    missing = get_missing_slots(label, sess.turns.slots)
    
    if missing:
        # Ask for the next missing slot based on priority
        priority_order = SLOT_PRIORITY.get(label, missing)
        next_missing = None
        for slot in priority_order:
            if slot in missing:
                next_missing = slot
                break
        
        if next_missing:
            return generate_slot_question(next_missing, lang, sess.turns.slots)
    
    # All required slots filled - generate confirmation
    return generate_confirmation(label, sess.turns.slots, lang)

# ------------------- ElevenLabs synth -------------------
def synthesize_polly_pcm(text:str, lang:str):
    try:
        l = (lang or "fr")[:2]
        voice_id = ELEVEN_VOICES.get(l) or ELEVEN_VOICES.get("fr") or ELEVEN_VOICES.get("en") or ELEVEN_VOICES.get("de") or ELEVEN_VOICES.get("it")
        if not _eleven or not ELEVEN_API_KEY or not voice_id:
            try:
                voice = VOICE_OVERRIDES.get(lang) or POLLY_VOICE.get(lang, POLLY_VOICE.get("fr"))
                return polly.synthesize_speech(
                    Text=text, OutputFormat="pcm", SampleRate="8000", VoiceId=voice, Engine="neural"
                )["AudioStream"].read()
            except Exception:
                return None

        audio_iter = _eleven.text_to_speech.convert(
            voice_id=voice_id,
            model_id=ELEVEN_MODEL_ID,
            text=text,
            optimize_streaming_latency=ELEVEN_LATENCY,
            output_format="pcm_16000",
        )

        if isinstance(audio_iter,(bytes,bytearray)):
            pcm16k = bytes(audio_iter)
        else:
            chunks=[]
            for chunk in audio_iter:
                chunks.append(chunk if isinstance(chunk,(bytes,bytearray)) else bytes(chunk))
            pcm16k = b"".join(chunks)

        if not pcm16k:
            return None
        pcm8k,_ = audioop.ratecv(pcm16k,2,1,16000,8000,None)
        return pcm8k
    except Exception as e:
        emit("error", err=f"elevenlabs_synthesize_error: {e}")
        return None

def tts_to_ulaw_frames(text:str, lang:str):
    pcm8k = synthesize_polly_pcm(text, lang)
    if not pcm8k:
        return []
    ulaw = audioop.lin2ulaw(pcm8k, 2)
    FRAME = 160
    return [base64.b64encode(ulaw[i:i+FRAME]).decode()
            for i in range(0, len(ulaw), FRAME) if ulaw[i:i+FRAME]]

async def stream_and_speak_response(ws: WebSocket, text: str, voice_id: str, call_state, tts_controller: TTSController):
    """
    Stream text-to-speech audio with interruption support.
    """
    # Early return if no text to speak
    if not text.strip():
        return True

    # Extract call identifiers
    stream_sid = call_state.get("twilio_stream_sid") if isinstance(call_state, dict) else getattr(call_state, 'twilio_stream_sid', None)
    
    # Log the start of TTS with truncated text for readability
    print(f"[TTS-SPEAK-{stream_sid}] AI ➔ {text[:60].replace(chr(10), ' ')}")

    # Create the TTS generator
    generator = _eleven.text_to_speech.stream(
        voice_id=voice_id,
        model_id=ELEVEN_MODEL_ID,
        text=text,
        optimize_streaming_latency=ELEVEN_LATENCY,
        output_format="pcm_16000"
    )

    # Configure the TTS controller
    tts_controller.is_speaking = True
    tts_controller.audio_queue = asyncio.Queue()
    tts_controller.current_generator = generator

    # Producer task to get audio chunks
    async def _produce_audio():
        try:
            for chunk in generator:
                await tts_controller.audio_queue.put(chunk)
            await tts_controller.audio_queue.put(None)
        except Exception as e:
            log.error(f"[TTS-PRODUCER-ERROR-{stream_sid}] {e}", exc_info=True)
            await tts_controller.audio_queue.put(None)

    # Start the producer task
    tts_controller.producer_task = asyncio.create_task(_produce_audio())
    
    try:
        # Process audio chunks
        while True:
            chunk = await tts_controller.audio_queue.get()
            if chunk is None:
                break
                
            # Convert to 8kHz μ-law and send
            pcm8k, _ = audioop.ratecv(chunk, 2, 1, 16000, 8000, None)
            ulaw = audioop.lin2ulaw(pcm8k, 2)
            
            # Split into Twilio-sized frames and send
            FRAME = 160
            for i in range(0, len(ulaw), FRAME):
                frame = ulaw[i:i+FRAME]
                if frame:
                    payload = base64.b64encode(frame).decode()
                    await ws.send(json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": payload}
                    }))
                    
        return True
    except asyncio.CancelledError:
        return False
    finally:
        tts_controller.is_speaking = False
        await ws.send(json.dumps({
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": "bot_done"}
        }))

async def send_tts(ws, sid, text, lang):
    """Wrapper around stream_and_speak_response for compatibility."""
    voice_id = ELEVEN_VOICES.get(lang[:2])
    if not voice_id:
        # Fallback to regular TTS
        for payload in tts_to_ulaw_frames(text, lang):
            await ws.send(json.dumps({"event":"media", "streamSid":sid, "media":{"payload":payload}}))
        await ws.send(json.dumps({"event":"mark", "streamSid":sid, "mark":{"name":"bot_done"}}))
    else:
        # Use new streaming TTS
        await stream_and_speak_response(
            ws=ws,
            text=text,
            voice_id=voice_id,
            call_state={"twilio_stream_sid": sid},
            tts_controller=tts_controller
        )

# --------- LLM Intent Classification ---------
INTENT_CLASSIFIER_PROMPT = """
You are an intent classifier for a clinic receptionist phone bot.
Classify the FINAL user message (considering the history) into exactly ONE label:

"appointment_booking" : schedule/reschedule/cancel.
"medical_advice"      : asks what THEY should do/take/avoid for THEIR situation, now
                        (e.g., "What should I do for my fever?", "Should I take ibuprofen?").
"medical_question"    : general/academic medical information not applied to their own case
                        (e.g., "What is appendicitis?", "How does an MRI work?", "What are statins?").
"critical"            : urgent emergency (severe chest pain, stroke signs, severe bleeding, etc.).
"suicidal"            : self-harm or suicide intent.
"violence"            : threats to others/the clinic.
"rude"                : insults/profanity aimed at agent/clinic.
"chitchat"            : small talk/off-topic.
"rag_query"           : clinic info (hours, team, insurance, policies).
"callback_request"    : requests for a callback or to be called back by a doctor/staff member
                        (e.g., "Can someone call me back?", "I need a callback", "Have doctor call me").
"speak_to_someone"    : wants to speak to or be connected to staff/doctor now
                        (e.g., "Can I talk to someone?", "Connect me to a doctor", "I want to speak to a nurse").
"prescription_renewal": asks to renew/refill a prescription or repeat medication
                        (e.g., "Please renew my prescription", "I need a refill", "ordonnance à renouveler").
"appointment_reschedule": wants to move/change an existing appointment to a new date/time.
"appointment_cancellation": wants to cancel an existing appointment.
"general"             : anything else, or replies that just move the flow along.

Rules:
- Base on the LAST user message with context.
- Prefer "medical_advice" when the user seeks a recommendation FOR THEMSELF.
- Prefer "medical_question" when it's informational or academic and not personal.
- Output STRICT JSON: {"label":"<one_of_above>","confidence":<0.0-1.0>}
- If unsure, choose "general".
"""


def _extract_json(s: str) -> Optional[dict]:
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
    except Exception:
        pass
    return None

def classify_intent_llm(text: str, history: str, lang: str) -> tuple[int, str]:
    print("[LLM Intent Classification Prompt]")
    clipped_text = text if len(text) <= INTENT_MAXLEN else text[:INTENT_MAXLEN]

    prompt_parts = [INTENT_CLASSIFIER_PROMPT]
    if history:
        print("History detected, including in prompt.")
        prompt_parts.append("--- Conversation History ---")
        prompt_parts.append(history)
        prompt_parts.append("---------------------------")

    prompt_parts.append(f"User message to classify: \"{clipped_text}\"")
    prompt_parts.append("JSON:")

    prompt = "\n".join(prompt_parts)
    print("Prompting LLAMA")
    raw = llama(prompt, lang)
    print("Raw LLM output:", raw)
    data = _extract_json(raw) or {}
    label = (data.get("label") or "general").strip().lower()
    if label not in LABEL2ID:
        label = "general"
    print("[LLM Intent Classification Result]", data)
    return LABEL2ID[label], label

# ----------- VKG / RAG Helpers ----------
def _normalize_list_or_str(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        out=[]
        for x in val:
            if x is None:
                continue
            if isinstance(x, str):
                out.append(x.strip())
            else:
                out.append(str(x).strip())
        return [s for s in out if s]
    return [str(val).strip()]

def _format_knowledge(hits: List[Dict[str, Any]], max_lines:int=12, include_assist:bool=True) -> str:
    def _as_lines(h):
        # prefer 'descriptions', then 'text', then 'content'
        for key in ("descriptions", "text", "content"):
            vals = h.get(key)
            if vals:
                for s in _normalize_list_or_str(vals):
                    yield f"- {s}"

    lines = []
    for h in hits:
        if len(lines) >= max_lines:
            break
        for d in _as_lines(h):
            if len(lines) >= max_lines:
                break
            lines.append(d)

        if include_assist:
            for a in _normalize_list_or_str(h.get("assist")):
                if len(lines) >= max_lines:
                    break
                lines.append(f"(➡ Step: {a})")

    return "\n".join(lines)

def _build_medical_question_prompt(lang:str, history:str, user_text:str, knowledge_text:str, max_words:int) -> str:
    # Neutral, informational, non-personal. No dosing/diagnosis.
    disclaimer = {
        "en": "Informational only. Not medical advice or diagnosis.",
        "fr": "À titre informatif. Pas un avis médical ni un diagnostic.",
        "de": "Nur zur Information. Keine medizinische Beratung oder Diagnose.",
        "it": "Solo informativo. Non è consulenza medica o diagnosi."
    }.get(lang, "Informational only. Not medical advice or diagnosis.")

    return (
        f"{SYSTEM_PROMPT}\n"
        f"{disclaimer}\n"
        f"Use ONLY the KNOWLEDGE below to briefly explain the concept in neutral terms.\n"
        f"- No dosing, no instructions tailored to the caller.\n"
        f"- If the user wants personal guidance, briefly offer booking at the end.\n\n"
        f"KNOWLEDGE:\n{knowledge_text or '– (no relevant facts found)'}\n\n"
        f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
        f"USER: {user_text}\n"
        f"ASSISTANT (under {max_words} words, sentence form — no bullet points). At the end, ask for their phone number if it's not already provided:"
    )

async def build_medical_question_rag_reply(user_text:str, history:str, lang:str, *, top_k:int=RAG_TOP_K) -> tuple[str, List[Dict[str, Any]]]:
    hits = await semantic_search(
        question=user_text,
        index_name=RAG_INDEX_NAME,
        doc_id=MEDICAL_ENCYC_DOC_ID,
        num_results=top_k,
    )
    knowledge_text = _format_knowledge(hits, max_lines=14, include_assist=False)
    # Allow a bit more room than default for definitions
    max_words = min(80, max(45, RESP_MAX_WORDS + 30))
    raw = llama(_build_medical_question_prompt(lang, history, user_text, knowledge_text, max_words), lang)
    reply = truncate_words(clean_reply(raw), max_words=max_words)
    return reply, hits


def _build_rag_prompt(lang:str, history:str, user_text:str, knowledge_text:str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n"
        f"You have ACCESS to KNOWLEDGE from the clinic database.\n"
        f"KNOWLEDGE (facts & steps):\n{knowledge_text or '– (no relevant facts found)'}\n\n"
        f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
        f"USER: {user_text}\n"
        f"ASSISTANT:\n"
        f"- Use ONLY the KNOWLEDGE above when giving factual info (hours, insurance, names, policies, steps).\n"
        f"- If not enough info, ask ONE brief clarifying question or offer to connect/book.\n"
        f"- Keep it under {RESP_MAX_WORDS} words, no bullet points in the final reply.\n"
        f"- Always ask for the caller's best phone number if it's not already provided."
    )

def _build_faq_advice_prompt(lang:str, history:str, user_text:str, knowledge_text:str) -> str:
    # Safer tone for medical advice from FAQ; never diagnose or prescribe.
    safety = {
        "en": "This is general guidance, not a diagnosis. If symptoms are severe or worsening, advise the emergency line. Offer appointment.",
        "fr": "Ce sont des conseils généraux, pas un diagnostic. Si les symptômes sont graves ou s'aggravent, recommandez la ligne d'urgence. Proposez un rendez-vous.",
        "de": "Dies sind allgemeine Hinweise, keine Diagnose. Bei schweren oder zunehmenden Symptomen die Notrufstelle empfehlen. Termin anbieten.",
        "it": "Indicazioni generali, non una diagnosi. Se i sintomi sono gravi o peggiorano, consiglia la linea di emergenza. Offrire appuntamento."
    }.get(lang, "This is general guidance, not a diagnosis. If symptoms are severe or worsening, advise the emergency line. Offer appointment.")

    return (
        f"{SYSTEM_PROMPT}\n"
        f"Use ONLY the FAQ KNOWLEDGE below to give short, generic guidance. No diagnosis or medication dosing.\n"
        f"{safety}\n"
        f"FAQ KNOWLEDGE:\n{knowledge_text or '– (no relevant FAQ found)'}\n\n"
        f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
        f"USER: {user_text}\n"
        f"ASSISTANT (concise, < {RESP_MAX_WORDS} words). At the end, ask for the caller's phone number to follow up, if not already provided:"
    )



async def generate_rag_reply(user_text:str, history:str, lang:str, *, top_k:int=RAG_TOP_K, doc_id:str=None) -> tuple[str, List[Dict[str, Any]]]:
    target_doc = doc_id or RAG_DOC_ID
    hits = await semantic_search(
        question=user_text,
        index_name=RAG_INDEX_NAME,
        doc_id=target_doc,
        num_results=top_k,
    )
    knowledge_text = _format_knowledge(hits, max_lines=14, include_assist=RAG_INCLUDE_ASSIST)
    # Use the standard RAG prompt by default
    raw = llama(_build_rag_prompt(lang, history, user_text, knowledge_text), lang)
    reply = truncate_words(clean_reply(raw), max_words=RESP_MAX_WORDS)
    return reply, hits

PREDEFINED_MEDICAL_QUESTION_MSG = {
    "en": "I can't provide general medical information over the phone. I can connect you with a doctor or help you book.",
    "fr": "Je ne peux pas fournir d'informations médicales générales par téléphone. Je peux vous mettre en relation avec un médecin ou fixer un rendez-vous.",
    "de": "Allgemeine medizinische Auskünfte gebe ich nicht am Telefon. Ich kann Sie mit einem Arzt verbinden oder einen Termin vereinbaren.",
    "it": "Non fornisco informazioni mediche generali al telefono. Posso metterti in contatto con un medico o fissare un appuntamento."
}

async def build_medical_advice_reply(user_text:str, history:str, lang:str) -> tuple[str, List[Dict[str, Any]]]:
    """
    Uses FAQ doc via RAG to give short, generic guidance (no diagnosis).
    """
    # Use the FAQ prompt for safer framing
    hits = await semantic_search(
        question=user_text,
        index_name=RAG_INDEX_NAME,
        doc_id=FAQ_DOC_ID,
        num_results=RAG_TOP_K,
    )
    knowledge_text = _format_knowledge(hits, max_lines=14, include_assist=RAG_INCLUDE_ASSIST)
    raw = llama(_build_faq_advice_prompt(lang, history, user_text, knowledge_text), lang)
    reply = truncate_words(clean_reply(raw), max_words=RESP_MAX_WORDS)
    return reply, hits


def build_medical_question_reply(user_text:str, history:str, lang:str) -> str:
    """
    Plays only a predefined message (no RAG).
    """
    return PREDEFINED_MEDICAL_QUESTION_MSG.get(lang, PREDEFINED_MEDICAL_QUESTION_MSG["en"])

# --------- Non-RAG Generators (Overflow paths) ---------
def build_simple_reply(user_text:str, history:str, lang:str) -> str:
    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
        f"USER: {user_text}\n"
        f"ASSISTANT (no external knowledge; keep under {RESP_MAX_WORDS} words). If not already collected, ask for their best phone number in case we need to follow up:"
    )
    return truncate_words(clean_reply(llama(prompt, lang)), max_words=RESP_MAX_WORDS)

def build_medical_reply(user_text:str, history:str, lang:str) -> str:
    boundary = {
        "en": "I cannot give medical advice. For that, you must speak with a doctor.",
        "fr": "Je ne peux pas donner de conseil médical. Pour cela, parlez à un médecin.",
        "de": "Ich kann keine medizinische Beratung geben. Dafür müssen Sie mit einem Arzt sprechen.",
        "it": "Non posso dare consigli medici. Per questo, parli con un medico."
    }.get(lang, "I cannot give medical advice. For that, you must speak with a doctor.")

    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"Your only goal is to state the boundary phrase and offer to book an appointment.\n"
        f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
        f"USER: {user_text}\n"
        f"ASSISTANT (start with '{boundary}'). At the end, ask for the caller's phone number so staff can follow up:"
    )
    raw = llama(prompt, lang)
    reply = truncate_words(clean_reply(raw), max_words=RESP_MAX_WORDS)
    if not reply.lower().startswith(boundary.lower()[:8]):
        reply = truncate_words(f"{boundary} {reply}", RESP_MAX_WORDS)
    return reply

def build_appointment_reply(user_text:str, history:str, lang:str, sess=None) -> str:
    """Wrapper for appointment booking - now uses orchestrator."""
    if sess:
        return truncate_words(build_orchestrated_reply("appointment_booking", user_text, history, lang, sess), max_words=RESP_MAX_WORDS)
    
    # Fallback to simplified prompt if no session
    prompt = (
        f"{SYS()}\n"
        f"Help with appointment booking. Ask for missing details one at a time.\n"
        f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
        f"USER: {user_text}\n"
        f"ASSISTANT:"
    )
    return truncate_words(clean_reply(llama(prompt, lang)), max_words=RESP_MAX_WORDS)

def build_reschedule_reply(user_text: str, history: str, lang: str, state: Optional[str], prev_date: Optional[str], prev_time: Optional[str], sess=None) -> tuple[str, str]:
    """Wrapper for appointment rescheduling - now uses orchestrator."""
    if sess:
        reply = truncate_words(build_orchestrated_reply("appointment_reschedule", user_text, history, lang, sess), max_words=RESP_MAX_WORDS)
        return reply, 'completed'
    
    # Fallback to simplified prompt if no session
    prompt = (
        f"{SYS()}\n"
        f"Help with appointment rescheduling. Ask for missing details one at a time.\n"
        f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
        f"USER: {user_text}\n"
        f"ASSISTANT:"
    )
    
    reply = truncate_words(clean_reply(llama(prompt, lang)), max_words=RESP_MAX_WORDS)
    return reply, 'completed'

def build_cancellation_reply(user_text: str, history: str, lang: str, state: Optional[str], sess=None) -> tuple[str, str]:
    """Wrapper for appointment cancellation - now uses orchestrator."""
    if sess:
        reply = truncate_words(build_orchestrated_reply("appointment_cancellation", user_text, history, lang, sess), max_words=RESP_MAX_WORDS)
        return reply, 'completed'
    
    # Fallback to simplified prompt if no session
    prompt = (
        f"{SYS()}\n"
        f"Help with appointment cancellation. Ask for missing details one at a time.\n"
        f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
        f"USER: {user_text}\n"
        f"ASSISTANT:"
    )
    
    reply = truncate_words(clean_reply(llama(prompt, lang)), max_words=RESP_MAX_WORDS)
    return reply, 'completed'

def _extract_name_from_text(user_text: str, history: str, lang: str) -> Optional[str]:
    try:
        schema = ["name"]
        prompt = (
            f"Extract the caller's personal name from the last user message, considering brief context.\n"
            f"Return STRICT JSON with these keys exactly: {schema}.\n"
            f"If unknown, set name to null. No extra text.\n\n"
            f"CONTEXT:\n{history or '(no prior turns)'}\n\n"
            f"USER: {user_text}\n"
            f"JSON:"
        )
        raw = llama(prompt, lang)
        data = _extract_json(raw) or {}
        name = data.get("name")
        if isinstance(name, str):
            name = name.strip()
        return name or None
    except Exception:
        return None

def build_prescription_renewal_reply(user_text: str, history: str, lang: str, state: Optional[str], sess=None) -> tuple[str, str, Optional[str]]:
    """
    Wrapper for prescription renewal - now uses orchestrator.
    Returns (reply, next_state, extracted_name)
    """
    if sess:
        reply = truncate_words(build_orchestrated_reply("prescription_renewal", user_text, history, lang, sess), max_words=RESP_MAX_WORDS)
        name = sess.turns.slots.get("patient_name")
        return reply, 'completed', name
    
    # Fallback to simplified prompt if no session
    prompt = (
        f"{SYS()}\n"
        f"Help with prescription renewal. Ask for missing details one at a time.\n"
        f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
        f"USER: {user_text}\n"
        f"ASSISTANT:"
    )
    
    reply = truncate_words(clean_reply(llama(prompt, lang)), max_words=RESP_MAX_WORDS)
    name = _extract_name_from_text(user_text, history, lang)
    
    return reply, 'completed', name

def store_prescription_renewal(stream_sid: str, phone_number: str, conversation_history: str, patient_name: Optional[str], summary: str, lang: str) -> bool:
    if not _r:
        print("Redis not available. Cannot store prescription renewal.")
        return False
    try:
        payload = {
            "status": "prescription_renewal",
            "phone_number": phone_number or "",
            "conversation_history": conversation_history or "",
            "patient_name": patient_name or "",
            "summary": summary or "",
            "description": summary or "",
            "language": lang or "",
            "timestamp": str(int(time.time())),
            "processed": "false"
        }
        _r.hset(stream_sid, mapping=payload)
        print(f"Prescription renewal stored for SID: {stream_sid}")
        return True
    except Exception as e:
        print(f"Error storing prescription renewal: {e}")
        return False

def extract_session_actions(conversation_history: str, lang: str) -> Dict[str, Any]:
    """Analyze the entire call and return a unified JSON of requested actions.

    Output schema (STRICT JSON):
    {
      "statuses": ["appointment_booking" | "prescription_renewal" | "callback_request" | "appointment_reschedule" | "appointment_cancellation", ...],
      "appointment": {
        "date_iso": "YYYY-MM-DD or null",
        "time_24h": "HH:MM or null",
        "timezone": "IANA tz or null",
        "day_of_week": "Monday..Sunday or null",
        "slot_start": "HH:MM or null",
        "slot_end": "HH:MM or null",
        "patient_status": "new|existing|null",
        "patient_name": "string or null",
        "appointment_type": "first_consultation|follow_up|iron_infusion|pragmafare|other_scheduling or null",
        "duration_minutes": "40|20|30 based on appointment type or null",
        "summary": "short sentence",
        "description": "short sentence",
        "notes": "string or null"
      } | null,
      "prescription": {
        "patient_name": "string or null",
        "summary": "short sentence",
        "description": "short sentence"
      } | null,
      "reschedule": {
        "prev_date_iso": "YYYY-MM-DD or null",
        "prev_time_24h": "HH:MM or null",
        "new_date_iso": "YYYY-MM-DD or null",
        "new_time_24h": "HH:MM or null",
        "appointment_type": "first_consultation|follow_up|iron_infusion|pragmafare|other_scheduling or null",
        "duration_minutes": "40|20|30 based on appointment type or null",
        "summary": "short sentence",
        "description": "short sentence"
      } | null,
      "cancellation": {
        "prev_date_iso": "YYYY-MM-DD or null",
        "prev_time_24h": "HH:MM or null",
        "summary": "short sentence",
        "description": "short sentence"
      } | null,
      "callback": {
        "requested": true|false,
        "summary": "short sentence",
        "description": "short sentence"
      } | null,
      "summary": "two short sentences overall"
    }

    Rules for reasoning:
    - Consider the full conversation context.
    - Include an action in statuses ONLY if the user actually wants it at the end. If they said never mind or cancelled it, EXCLUDE it.
    - Fill objects only for included statuses; set others to null.
    - Keep text concise. Do not add commentary outside JSON.
    """
    try:
        from datetime import datetime
        now = datetime.now()
        today_iso = now.strftime("%Y-%m-%d")
        prompt = (
            f"You are extracting final requested actions from a phone conversation.\n"
            f"Today's date is {today_iso}. Language code is {lang}.\n"
            f"Return STRICT JSON only with keys exactly as documented.\n"
            f"Determine which of these the caller ultimately wants (not cancelled): appointment_booking, prescription_renewal, callback_request, appointment_reschedule, appointment_cancellation.\n"
            f"For appointments, extract the appointment_type from these options:\n"
            f"- first_consultation (40 minutes)\n"
            f"- follow_up (20 minutes)\n"
            f"- iron_infusion (20 minutes)\n"
            f"- pragmafare (30 minutes)\n"
            f"- other_scheduling (20 minutes)\n"
            f"Set duration_minutes to 40, 20, or 30 based on the appointment_type.\n"
            f"When reschedule/cancellation applies, also provide 'reschedule' and/or 'cancellation' objects with keys as in the schema.\n"
            f"Conversation History:\n{conversation_history}\n\nJSON:"
        )
        raw = llama(prompt, lang)
        data = _extract_json(raw) or {}
        # Ensure structure and keys exist
        out: Dict[str, Any] = {}
        out["statuses"] = data.get("statuses") or []
        out["appointment"] = data.get("appointment") if isinstance(data.get("appointment"), dict) else None
        out["prescription"] = data.get("prescription") if isinstance(data.get("prescription"), dict) else None
        out["callback"] = data.get("callback") if isinstance(data.get("callback"), dict) else None
        out["reschedule"] = data.get("reschedule") if isinstance(data.get("reschedule"), dict) else None
        out["cancellation"] = data.get("cancellation") if isinstance(data.get("cancellation"), dict) else None
        out["summary"] = data.get("summary") or "Session actions extracted."
        # Normalize statuses strings
        try:
            out["statuses"] = [str(s).strip().lower() for s in (out["statuses"] or []) if s]
        except Exception:
            out["statuses"] = []
        return out
    except Exception as e:
        print(f"Error extracting session actions: {e}")
        return {"statuses": [], "appointment": None, "prescription": None, "callback": None, "summary": "Extraction error"}


def store_session_actions(stream_sid: str, phone_number: str, conversation_history: str, actions: Dict[str, Any], lang: str) -> bool:
    """Store unified session actions in Redis under the session SID key."""
    if not _r:
        print("Redis not available. Cannot store session actions.")
        return False
    try:
        payload = {
            "status_list": json.dumps(actions.get("statuses") or [], ensure_ascii=False),
            "phone_number": phone_number or "",
            "conversation_history": conversation_history or "",
            "appointment_json": json.dumps(actions.get("appointment") or {}, ensure_ascii=False),
            "prescription_json": json.dumps(actions.get("prescription") or {}, ensure_ascii=False),
            "callback_json": json.dumps(actions.get("callback") or {}, ensure_ascii=False),
            "reschedule_json": json.dumps(actions.get("reschedule") or {}, ensure_ascii=False),
            "cancellation_json": json.dumps(actions.get("cancellation") or {}, ensure_ascii=False),
            "summary": actions.get("summary") or "",
            "language": lang or "",
            "timestamp": str(int(time.time())),
            "processed": "false"
        }
        _r.hset(stream_sid, mapping=payload)
        print(f"Unified session actions stored for SID: {stream_sid}")
        return True
    except Exception as e:
        print(f"Error storing session actions: {e}")
        return False


def extract_appointment_details(conversation_history: str, lang: str) -> Dict[str, Any]:
    """Run an LLM pass at session end to extract appointment details as strict JSON.

    Returns a dictionary with keys like date_iso, time_24h, timezone, day_of_week,
    slot_start, slot_end, patient_status, patient_name, summary, description, notes.
    Unknown fields should be null.
    """
    try:
        # Compute today's context
        from datetime import datetime
        now = datetime.now()
        today_iso = now.strftime("%Y-%m-%d")
        weekday = now.strftime("%A")  # e.g., Monday

        schema_hint = {
            "date_iso": "YYYY-MM-DD or null",
            "time_24h": "HH:MM in 24h or null",
            "timezone": "IANA tz like Europe/Zurich or null",
            "day_of_week": "Monday..Sunday or null",
            "slot_start": "HH:MM 24h if a time range was agreed, else null",
            "slot_end": "HH:MM 24h if a time range was agreed, else null",
            "patient_status": "new|existing|null",
            "patient_name": "string or null",
            "appointment_type": "first_consultation|follow_up|iron_infusion|pragmafare|other_scheduling or null",
            "duration_minutes": "40|20|30 based on appointment type or null",
            "summary": "1-2 sentences summary of what was agreed",
            "description": "same as summary or slightly more detail",
            "notes": "any extra constraints mentioned (e.g., morning only) or null"
        }

        prompt = (
            f"You are extracting appointment booking data from a phone conversation.\n"
            f"Today's date is {today_iso} and today is {weekday}. The caller's language code is {lang}.\n"
            f"Return STRICT JSON only, with these keys exactly: {list(schema_hint.keys())}.\n"
            f"- Use 24-hour time.\n"
            f"- If the user specifies a weekday (e.g., Tuesday), compute the absolute calendar date (date_iso) as the NEXT occurrence of that day relative to today (do NOT default to today unless they explicitly said 'today'). Example: if today is Monday 2025-09-22 and they say Tuesday, set date_iso to 2025-09-23.\n"
            f"- If both a specific date and a weekday are present and they disagree, prefer the explicit date.\n"
            f"- If a field is unknown or not stated, set it to null.\n"
            f"- Do not add extra text outside JSON.\n\n"
            f"Conversation History:\n{conversation_history}\n\n"
            f"JSON:"
        )

        raw = llama(prompt, lang)
        data = _extract_json(raw) or {}

        # Ensure only expected keys, fill missing with None
        expected_keys = [
            "date_iso","time_24h","timezone","day_of_week",
            "slot_start","slot_end","patient_status","patient_name",
            "appointment_type","duration_minutes","summary","description","notes"
        ]
        cleaned: Dict[str, Any] = {k: (data.get(k, None)) for k in expected_keys}

        # Helper: map weekday string to index (Mon=0..Sun=6) for English basic support
        def _weekday_index_en(name: str) -> Optional[int]:
            if not isinstance(name, str):
                return None
            nm = name.strip().lower()
            names = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
            return names.index(nm) if nm in names else None

        # Detect explicit weekday mention in conversation (English only fallback)
        mentioned_idx: Optional[int] = None
        try:
            wk_rx = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
            m = wk_rx.search(conversation_history or "")
            if m:
                mentioned_idx = _weekday_index_en(m.group(1))
        except Exception:
            mentioned_idx = None

        # Compute next date for a given weekday index from 'now'
        def _next_date_for_weekday(target_idx: int) -> str:
            from datetime import timedelta
            today_idx = now.weekday()  # Mon=0..Sun=6
            delta = (target_idx - today_idx) % 7
            # If same day mentioned but not explicitly "today", prefer next week only if text says "next"; we keep same week when delta==0 and 'today' is mentioned.
            # Since detection of 'today' is language-specific, we keep the next occurrence rule only when delta>0; if delta==0 and no 'today' in text, use today.
            if delta == 0:
                # try to detect explicit 'today'
                if re.search(r"\btoday\b", (conversation_history or "").lower()):
                    add_days = 0
                else:
                    add_days = 7  # assume next week if they didn't say 'today'
            else:
                add_days = delta
            return (now.date() + timedelta(days=add_days)).strftime("%Y-%m-%d")

        # Reconcile day/date from LLM output and conversation mention
        try:
            # If LLM provided day_of_week but date_iso missing or mismatched, fix date_iso
            day_name = cleaned.get("day_of_week")
            llm_day_idx = _weekday_index_en(day_name) if day_name else None
            date_iso = cleaned.get("date_iso")

            if mentioned_idx is not None:
                # Prefer the mentioned weekday from the conversation
                fixed_date = _next_date_for_weekday(mentioned_idx)
                cleaned["date_iso"] = fixed_date
                cleaned["day_of_week"] = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][mentioned_idx]
            elif llm_day_idx is not None and (not date_iso or True):
                # If LLM gave a day but no/any date, compute from today for consistency
                fixed_date = _next_date_for_weekday(llm_day_idx)
                cleaned["date_iso"] = fixed_date

            # Backfill day_of_week from date_iso if still missing
            if cleaned.get("date_iso") and not cleaned.get("day_of_week"):
                from datetime import datetime as _dt
                dt = _dt.strptime(str(cleaned["date_iso"]), "%Y-%m-%d")
                cleaned["day_of_week"] = dt.strftime("%A")
        except Exception:
            pass

        # Minimal summary/description if missing
        if not cleaned.get("summary"):
            cleaned["summary"] = "Appointment booking details extracted from call."
        if not cleaned.get("description"):
            cleaned["description"] = cleaned.get("summary")

        return cleaned
    except Exception as e:
        print(f"Error extracting appointment details: {e}")
        return {}

def store_appointment_booking(stream_sid: str, phone_number: str, conversation_history: str, appointment: Dict[str, Any], lang: str) -> bool:
    """Store appointment booking info in Redis under the session SID key."""
    if not _r:
        print("Redis not available. Cannot store appointment booking.")
        return False

def store_reschedule(stream_sid: str, phone_number: str, conversation_history: str, reschedule: Dict[str, Any], lang: str) -> bool:
    if not _r:
        print("Redis not available. Cannot store reschedule.")
        return False
    try:
        payload = {
            "status": "appointment_reschedule",
            "phone_number": phone_number or "",
            "conversation_history": conversation_history or "",
            "reschedule_json": json.dumps(reschedule or {}, ensure_ascii=False),
            "summary": (reschedule or {}).get("summary") or "",
            "description": (reschedule or {}).get("description") or "",
            "language": lang or "",
            "timestamp": str(int(time.time())),
            "processed": "false"
        }
        _r.hset(stream_sid, mapping=payload)
        print(f"Reschedule stored for SID: {stream_sid}")
        return True
    except Exception as e:
        print(f"Error storing reschedule: {e}")
        return False

def store_cancellation(stream_sid: str, phone_number: str, conversation_history: str, cancellation: Dict[str, Any], lang: str) -> bool:
    if not _r:
        print("Redis not available. Cannot store cancellation.")
        return False
    try:
        payload = {
            "status": "appointment_cancellation",
            "phone_number": phone_number or "",
            "conversation_history": conversation_history or "",
            "cancellation_json": json.dumps(cancellation or {}, ensure_ascii=False),
            "summary": (cancellation or {}).get("summary") or "",
            "description": (cancellation or {}).get("description") or "",
            "language": lang or "",
            "timestamp": str(int(time.time())),
            "processed": "false"
        }
        _r.hset(stream_sid, mapping=payload)
        print(f"Cancellation stored for SID: {stream_sid}")
        return True
    except Exception as e:
        print(f"Error storing cancellation: {e}")
        return False

    try:
        payload = {
            "status": "appointment_booking",
            "phone_number": phone_number or "",
            "conversation_history": conversation_history or "",
            "appointment_json": json.dumps(appointment or {}, ensure_ascii=False),
            "summary": (appointment or {}).get("summary") or "",
            "description": (appointment or {}).get("description") or "",
            "language": lang or "",
            "timestamp": str(int(time.time())),
            "processed": "false"
        }
        _r.hset(stream_sid, mapping=payload)
        print(f"Appointment booking stored for SID: {stream_sid}")
        return True
    except Exception as e:
        print(f"Error storing appointment booking: {e}")
        return False

def build_callback_reply(user_text: str, history: str, lang: str, callback_state: str = None, sess=None) -> tuple[str, str]:
    """
    Build callback flow responses using orchestrator when possible.
    Returns tuple of (reply, next_callback_state)
    """
    # If we have a session and this is the initial callback request, use orchestrator
    if sess and callback_state is None:
        reply = truncate_words(build_orchestrated_reply("callback_request", user_text, history, lang, sess), max_words=RESP_MAX_WORDS)
        return reply, 'completed'
    if callback_state is None:
        # Initial callback request - check availability
        availability_prompt = (
            f"{SYSTEM_PROMPT}\n"
            f"The user is requesting a callback. You should:\n"
            f"1. Say 'wait a second please, let me check if anyone is available'\n"
            f"2. Then immediately say 'sorry, no one is available right now'\n"
            f"3. Then offer: 'but I can request a callback for you if that works'\n"
            f"Keep it natural and conversational, under {RESP_MAX_WORDS} words.\n\n"
            f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
            f"USER: {user_text}\n"
            f"ASSISTANT:"
        )
        reply = truncate_words(clean_reply(llama(availability_prompt, lang)), max_words=RESP_MAX_WORDS)
        return reply, 'offered'
    
    elif callback_state == 'offered':
        # User responded to callback offer
        user_lower = user_text.lower()
        
        # Check for positive responses
        positive_indicators = {
            "en": ["yes", "yeah", "sure", "ok", "okay", "please", "that works", "sounds good"],
            "fr": ["oui", "ouais", "d'accord", "ok", "okay", "s'il vous plaît", "ça marche", "c'est bon"],
            "de": ["ja", "okay", "ok", "bitte", "das geht", "das passt", "gerne"],
            "it": ["sì", "va bene", "ok", "okay", "per favore", "funziona", "bene"]
        }
        
        negative_indicators = {
            "en": ["no", "nah", "not", "don't", "never mind"],
            "fr": ["non", "pas", "jamais", "laisse tomber"],
            "de": ["nein", "nicht", "lass mal", "vergiss es"],
            "it": ["no", "non", "mai", "lascia perdere"]
        }
        
        lang_key = lang[:2] if lang else "en"
        is_positive = any(indicator in user_lower for indicator in positive_indicators.get(lang_key, positive_indicators["en"]))
        is_negative = any(indicator in user_lower for indicator in negative_indicators.get(lang_key, negative_indicators["en"]))
        
        if is_positive and not is_negative:
            # User wants callback - confirm and ask for phone
            confirm_prompt = (
                f"{SYSTEM_PROMPT}\n"
                f"The user agreed to the callback request. You should:\n"
                f"1. Acknowledge their agreement positively\n"
                f"2. Ask for their phone number to arrange the callback\n"
                f"Keep it brief and professional, under {RESP_MAX_WORDS} words.\n\n"
                f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
                f"USER: {user_text}\n"
                f"ASSISTANT:"
            )
            reply = truncate_words(clean_reply(llama(confirm_prompt, lang)), max_words=RESP_MAX_WORDS)
            return reply, 'waiting_phone'
        
        elif is_negative:
            # User declined callback
            decline_prompt = (
                f"{SYSTEM_PROMPT}\n"
                f"The user declined the callback offer. You should:\n"
                f"1. Acknowledge their response politely\n"
                f"2. Ask if there's anything else you can help with\n"
                f"Keep it brief and helpful, under {RESP_MAX_WORDS} words.\n\n"
                f"CONVERSATION HISTORY:\n{history or '(no prior turns)'}\n\n"
                f"USER: {user_text}\n"
                f"ASSISTANT:"
            )
            reply = truncate_words(clean_reply(llama(decline_prompt, lang)), max_words=RESP_MAX_WORDS)
            return reply, None
        
        else:
            # Unclear response - clarify
            clarify_messages = {
                "en": "Would you like me to request a callback for you? Please say yes or no.",
                "fr": "Voulez-vous que je demande un rappel pour vous ? Dites oui ou non s'il vous plaît.",
                "de": "Möchten Sie, dass ich einen Rückruf für Sie anfordere? Bitte sagen Sie ja oder nein.",
                "it": "Vuoi che richieda una richiamata per te? Per favore dimmi sì o no."
            }
            reply = clarify_messages.get(lang_key, clarify_messages["en"])
            return reply, 'offered'
    
    elif callback_state == 'waiting_phone':
        # Extract phone number and process callback
        phone_pattern = re.compile(r'[\d\s\-\(\)\.+]{10,}')
        phone_match = phone_pattern.search(user_text)
        
        if phone_match:
            phone_number = re.sub(r'[^\\d+]', '', phone_match.group())
            confirmation_messages = {
                "en": f"Perfect! I've recorded your callback request. Someone will call you back as soon as possible.",
                "fr": f"Parfait ! J'ai enregistré votre demande de rappel. Quelqu'un vous rappellera dès que possible.",
                "de": f"Perfekt! Ich habe Ihre Rückruf-Anfrage aufgezeichnet. Jemand wird Sie so schnell wie möglich zurückrufen.",
                "it": f"Perfetto! Ho registrato la tua richiesta di richiamata. Qualcuno ti richiamerà il prima possibile."
            }
            reply = confirmation_messages.get(lang_key, confirmation_messages["en"])
            return reply, 'completed'
        else:
            # No valid phone number found
            phone_request_messages = {
                "en": "I didn't catch a valid phone number. Could you please provide your phone number for the callback?",
                "fr": "Je n'ai pas saisi un numéro valide. Pouvez-vous donner votre numéro de téléphone pour le rappel ?",
                "de": "Ich habe keine gültige Telefonnummer verstanden. Können Sie bitte Ihre Telefonnummer für den Rückruf angeben?",
                "it": "Non ho capito un numero di telefono valido. Puoi fornire il tuo numero per la richiamata?"
            }
            reply = phone_request_messages.get(lang_key, phone_request_messages["en"])
            return reply, 'waiting_phone'
    
    # Default fallback
    return build_simple_reply(user_text, history, lang), None


# ----------------- Segment-based VAD Parameters -----------------
SPEECH_THRESHOLD = 0.25   # lower = keep recording longer
MIN_SPEECH_FRAMES = 10    # ~320ms minimum speech before valid
MAX_SILENCE_FRAMES = 15   # ~480ms silence required to cut
# ----------------- WebSocket Handler -----------------

async def handle(ws):
    sess = Session(caller(ws)); sess.turns = TurnManager()
    stream_sid = None
    audio_processor = None
    # sess.current_segment_task = None
    loop = asyncio.get_running_loop()

    # async def _cancel_all_tasks():
    #     print("[Cleanup] Cancelling all running tasks.")
    #     t = sess.current_segment_task
    #     if t and not t.done():
    #         t.cancel()
    #         try:
    #             await t
    #         except asyncio.CancelledError:
    #             pass
    #     sess.current_segment_task = None

    #     # Stop TTS
    #     await tts_controller.stop_immediately()

    #     # Make finish_bot_turn idempotent or guard it (see below)
    #     try:
    #         sess.turns.finish_bot_turn()
    #     except Exception as e:
    #         print("finish_bot_turn error:", e)

    async def interrupt_if_user_speaking():
        """Handle user interruption by stopping TTS and clearing audio."""
        try:
            log.warning("[INTERRUPT] 🚨 USER SPEECH DETECTED! Interrupting TTS immediately.")
            if tts_controller:
                await tts_controller.stop_immediately()
                log.info("[INTERRUPT] ✅ TTS stopped successfully.")
            # Clear queued audio on Twilio side to stop playback
            if stream_sid and ws:
                try:
                    await ws.send(json.dumps({
                        "event": "clear",
                        "streamSid": stream_sid
                    }))
                    log.info("[INTERRUPT] ✅ Twilio audio cleared successfully.")
                except Exception as e:
                    log.warning(f"[INTERRUPT] ❌ Failed to clear Twilio audio: {e}")
        except Exception as e:
            log.error(f"[INTERRUPT] ❌ Error in interrupt check: {e}")

    def handle_interruption():
        print("[Interruption] VAD detected speech. Cancelling all bot actions.")
        # asyncio.run_coroutine_threadsafe(_cancel_all_tasks(), loop)
        asyncio.run_coroutine_threadsafe(interrupt_if_user_speaking(), loop)

    ### CALLBACK 2: For Inactivity ###
    async def handle_inactivity():
        print(f"[Inactivity] VAD detected silence for {INACTIVITY_TIMEOUT}s. Sending prompt.")
    #     lang = sess.turns.lang or "en"
    #     prompt_text = {"en": "Are you still there?", "fr": "Êtes-vous toujours là?", "de": "Sind Sie noch da?", "it": "Ci sei ancora?"}.get(lang, "Are you still there?")
        
    #     emit("inactive_prompt", lang=lang)
        
    #     # Ensure we don't interrupt ourselves
    #     if sess.current_tts_task and not sess.current_tts_task.done():
    #         sess.current_tts_task.cancel()
        
    #     # The bot takes its turn to send the prompt
    #     await process_inactivity_prompt(ws, stream_sid, prompt_text, lang, sess)

    try:
        async for raw in ws:
            ev = json.loads(raw)

            # Stream start/change
            if ev.get("event") == "start" and 'streamSid' in ev and stream_sid != ev['streamSid']:
                stream_sid = ev['streamSid']
                # Try to extract caller phone from Twilio start event customParameters or fields
                start_info = ev.get("start", {}) if isinstance(ev.get("start", {}), dict) else {}
                # Resolve using customParameters, else REST lookup via callSid
                from_num = await resolve_twilio_caller_number(start_info)
                # Final fallback to header/query parsing if everything else failed
                if not from_num:
                    fallback = caller(ws)
                    from_num = _normalize_phone_number(fallback)
                # Attach to session for later storage use
                try:
                    sess.caller_phone = from_num
                except Exception:
                    pass

                print(f"Caller connected, starting session {stream_sid}")
                emit("start", sid=stream_sid, frm=from_num)
                audio_processor = AudioPipeline(
                    session_id=stream_sid,
                    on_speech_started=handle_interruption,
                    on_inactivity=handle_inactivity,
                    inactivity_timeout=INACTIVITY_TIMEOUT
                )

            # Media payload
            if ev.get("event") == "media":
                if not audio_processor: continue
                
                # print("Received media frame", flush=True)
                result = await audio_processor._run_audio_processing(base64.b64decode(ev['media']['payload']))
                if result:
                    print("Result from faster whisper: ", result)
                    await process_segment(ws, stream_sid, result, sess)

            # Stop or disconnect
            elif ev.get("event") in ("stop", "disconnect"):
                print("Caller disconnected")
                break

    except websockets.exceptions.ConnectionClosedOK:
        print("Connection closed normally.")
    except Exception as e:
        emit("error", err=str(e))
    finally:
        print("Session ended.")
        # On session end, extract and store unified actions
        try:
            if sess:
                try:
                    lang = (sess.turns.lang or "fr")
                except Exception:
                    lang = "fr"

                # Build conversation history text
                history_text = sess.turns.build_context_text()

                # Prefer end-user phone from conversation or session/start headers
                try:
                    # Try to pull the last user utterance from the session history
                    last_user_utterance = None
                    try:
                        for e in reversed(list(sess.turns.events)):
                            ut = e.get("user")
                            if ut:
                                last_user_utterance = ut
                                break
                    except Exception:
                        last_user_utterance = None
                    inline_phone = extract_phone_from_text(last_user_utterance) if last_user_utterance else None
                    phone_number = inline_phone or getattr(sess, 'caller_phone', None) or caller(ws) or ""
                except Exception:
                    phone_number = ""

                # Unified extraction
                actions = extract_session_actions(history_text, lang)

                # Store unified result in Redis under the session SID
                if stream_sid:
                    success = store_session_actions(stream_sid, phone_number, history_text, actions, lang)
                    if success:
                        emit("session_actions_stored", phone=phone_number or "missing", statuses=actions.get("statuses") or [])
        except Exception as e2:
            print(f"Error during session-end unified processing: {e2}")

async def process_segment(ws, sid, res, sess):
    print("Processing segment result:", res)
    sess.turns.start_bot_turn()
    print(f"[SESSION {sid}] Detected text: '{res.get('text')}' with language '{res.get('language')}'")
    #try:
    text = (res.get("text") or "").strip()
    # discard dubious segments
    # if res.get("o_speech_prob", 0.0) > 0.4 or not text:
    #     return

    lang = res.get("language") or "fr"
    # TODO: lang = choose_lang(sess.turns.lang, whisper_lang, text)
    sess.turns.lang = lang
    emit("stt", lang=lang, text=text)
    print(f"[SESSION {sid}] Final text: '{text}' | lang={lang}")
    # Quick exit on GOODBYE
    if GOODBYE.search(text):
        farewell = {
            "en": "Goodbye.",
            "fr": "Au revoir.",
            "de": "Auf Wiedersehen.",
            "it": "Arrivederci.",
        }.get(lang, "Goodbye.")
        emit("goodbye", lang=lang)
        await send_tts(ws, sid, farewell, lang)
        return

    # Safety/policy: rude language fast-path
    rude = any(w in text.lower() for w in BAD_WORDS.get(lang, set()))
    if rude:
        class_id, label = LABEL2ID["rude"], "rude"
        print(f"[Rude Fast-Path] text='{text}' | class_id={class_id} | label={label}")
    else:
        print("[LLM Intent Classification]")
        history = sess.turns.build_context_text()
        class_id, label = classify_intent_llm(text, history, lang)
        print(f"[LLM DECISION] text='{text}' | class_id={class_id} | label={label}")
        emit("llm_decision", text=text, class_id=class_id, label=label)

    # -------- Safety/Policy & Small-talk quick replies ----------
    quick_replies = {
        "critical": {
            "en": "This is an emergency. Please hang up and call the emergency line immediately.",
            "fr": "Ceci est une urgence. Raccrochez et appelez la ligne d'urgence immédiatement.",
            "de": "Das ist ein Notfall. Bitte legen Sie auf und rufen Sie sofort die Notrufstelle an.",
            "it": "È un'emergenza. Riaggancia e chiama subito la linea di emergenza.",
        },
        "threat": {
            "en": "Threats are not tolerated. Goodbye.",
            "fr": "Les menaces ne sont pas tolérées. Au revoir.",
            "de": "Drohungen werden nicht toleriert. Auf Wiedersehen.",
            "it": "Le minacce non sono tollerate. Arrivederci.",
        },
        "polite": {
            "en": "Please remain polite. How can I help?",
            "fr": "Veuillez rester poli. Comment puis-je vous aider ?",
            "de": "Bitte bleiben Sie höflich. Wie kann ich helfen?",
            "it": "Per favore, sii gentile. Come posso aiutarti?",
        },
        "scope": {
            "en": "I can only help with clinic-related questions.",
            "fr": "Je peux seulement aider pour des questions liées à la clinique.",
            "de": "Ich kann nur bei klinikbezogenen Fragen helfen.",
            "it": "Posso aiutare solo con domande relative alla clinica.",
        },
        "rude": {  # matches LABEL2ID["rude"]
            "en": "Please be respectful. How can I help with your clinic request?",
            "fr": "Merci de rester respectueux. Comment puis-je aider pour votre demande liée à la clinique ?",
            "de": "Bitte bleiben Sie respektvoll. Wobei kann ich in Bezug auf die Klinik helfen?",
            "it": "Per favore, sii rispettoso. Come posso aiutarti riguardo alla clinica?",
        },
        "suicidal": {
            "en": "I'm concerned for your safety. Please hang up and call the emergency line now.",
            "fr": "Je suis inquiet pour votre sécurité. Raccrochez et appelez la ligne d'urgence immédiatement.",
            "de": "Ich mache mir Sorgen um Ihre Sicherheit. Bitte legen Sie auf und rufen Sie jetzt die Notrufstelle an.",
            "it": "Sono preoccupato per la tua sicurezza. Riaggancia e chiama subito la linea di emergenza."
        },
        "violence": {
            "en": "Threats are not tolerated. Goodbye.",
            "fr": "Les menaces ne sont pas tolérées. Au revoir.",
            "de": "Drohungen werden nicht toleriert. Auf Wiedersehen.",
            "it": "Le minacce non sono tollerate. Arrivederci."
        },
    }

    reply = None
    if label in quick_replies:
        reply = quick_replies[label].get(lang)

    # Rebuild history just before routing if needed
    history = sess.turns.build_context_text()

    # -------- INTENT ROUTING (VKG only when rag_query) ----------
    if reply is None and label == "rag_query":
        rag_reply, hits = await generate_rag_reply(text, history, lang, top_k=RAG_TOP_K, doc_id=RAG_DOC_ID)
        reply = rag_reply
        emit("rag", hits=hits)

    elif reply is None and label == "appointment_booking":
        reply = build_appointment_reply(text, history, lang, sess)    # Now uses orchestrator
        try:
            # Mark that appointment booking occurred in this session
            sess.turns.appointment_detected = True
        except Exception:
            pass

    elif reply is None and label == "callback_request":
        # Handle callback request using orchestrator; storage deferred to end of session
        reply, new_callback_state = build_callback_reply(text, history, lang, sess.turns.callback_state, sess)
        sess.turns.callback_state = new_callback_state

    elif reply is None and label == "appointment_reschedule":
        # Simple reschedule handling using orchestrator
        reply, _ = build_reschedule_reply(text, history, lang, None, None, None, sess)

    elif reply is None and label == "appointment_cancellation":
        # Simple cancellation handling using orchestrator
        reply, _ = build_cancellation_reply(text, history, lang, None, sess)

    elif reply is None and label == "speak_to_someone":
        # Availability flow → then offer callback; manage state only, storage deferred
        reply, new_callback_state = build_callback_reply(text, history, lang, sess.turns.callback_state, sess)
        sess.turns.callback_state = new_callback_state
        if new_callback_state == 'completed':
            # No immediate storage here; unified storage at session end
            pass

    elif reply is None and label == "medical_advice":
        reply, hits = await build_medical_advice_reply(text, history, lang)  # FAQ + VKG
        emit("faq_rag", hits=hits)

    elif reply is None and label == "medical_question":
        reply, hits = await build_medical_question_rag_reply(text, history, lang)  # <— RAG path
        emit("medical_rag", hits=hits)
           # PREDEFINED ONLY

    elif reply is None and label == "rude":
        # already covered by quick_replies above, but keep as fallback safety
        reply = quick_replies["rude"].get(lang)

    elif reply is None and label == "prescription_renewal":
        # Handle prescription renewal flow using orchestrator
        pr_reply, new_state, extracted_name = build_prescription_renewal_reply(text, history, lang, sess.turns.prescription_state, sess)
        reply = pr_reply
        sess.turns.prescription_state = new_state
        if new_state == 'completed' or extracted_name:
            # Store in Redis
            try:
                phone_pattern = re.compile(r"[\d\s\-\(\)\.+]{7,}")
                phone_match = phone_pattern.search(text)
                phone_number = re.sub(r"[^\d+]", "", phone_match.group()) if phone_match else (getattr(sess, 'caller_phone', None) or caller(ws) or "")
            except Exception:
                phone_number = getattr(sess, 'caller_phone', None) or caller(ws) or ""
            base_history = sess.turns.build_context_text()
            history_with_current = (base_history + "\n" if base_history else "") + f"User: {text}"
            summary = generate_conversation_summary(history_with_current, lang)
            success = store_prescription_renewal(sid, phone_number, history_with_current, extracted_name, summary, lang)
            if success:
                emit("prescription_renewal_stored", phone=phone_number or "missing", name=extracted_name or "", summary=summary)

    # Handle callback state responses even if not classified as callback_request
    elif reply is None and sess.turns.callback_state in ['offered', 'waiting_phone']:
        reply, new_callback_state = build_callback_reply(text, history, lang, sess.turns.callback_state, sess)
        sess.turns.callback_state = new_callback_state
        
        # If callback is completed, do not store immediately (defer to session end)
        if new_callback_state == 'completed':
            pass

    # Handle prescription renewal state even if not classified
    elif reply is None and sess.turns.prescription_state in ['waiting_name']:
        pr_reply, new_state, extracted_name = build_prescription_renewal_reply(text, history, lang, sess.turns.prescription_state, sess)
        reply = pr_reply
        sess.turns.prescription_state = new_state

    elif reply is None:
        reply = build_simple_reply(text, history, lang)

    if not reply:
        reply = build_simple_reply(text, history, lang)

    # Log + emit + speak
    sess.turns.log_pair(user_text=text, bot_text=reply, class_id=class_id, label=label)
    if hasattr(sess, "log"):
        sess.log(user=text, bot=reply, class_id=class_id)

    emit("bot", lang=lang, reply=reply)

    # Custom flag example (from your screenshot)
    sess.turns.awaiting_doctor = ("Cyrus Terrani" in reply and "Riddley Auguste" in reply)

    await send_tts(ws, sid, reply, lang)
    sess.turns.finish_bot_turn()


    # except Exception as e:
    #     emit("error", err=str(e))
    #     raise
    # finally:
    #     sess.turns.finish_bot_turn()

    print(f"Websocket listening on :{LISTEN_PORT}")

# async def process_inactivity_prompt(ws, sid, text, lang, sess):
#     """Sends the inactivity TTS prompt and manages the session state."""
#     sess.turns.start_bot_turn()
#     tts_task = asyncio.create_task(send_tts(ws, sid, text, lang))
#     sess.running_tasks.append(tts_task)

#     def tts_done_callback(task):
#         sess.turns.finish_bot_turn() # Always finish the turn
#         if sess.current_tts_task is task:
#             sess.current_tts_task = None
            
#     tts_task.add_done_callback(tts_done_callback)

async def main():
    print(f"Whisper model in use: {WHISPER_MODEL}")
    async with serve(handle, LISTEN_HOST, LISTEN_PORT, ping_interval=None):
        await asyncio.Future()  

if __name__ == "__main__":

    asyncio.run(main())



