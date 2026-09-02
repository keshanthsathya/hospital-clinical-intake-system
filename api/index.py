import os
import json
import traceback
import urllib.request
import urllib.error
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="Clinical Intake & Triage System",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# CORS middleware - allow all origins so frontends can communicate easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler to give crystal-clear JSON diagnostics
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": str(exc),
            "type": type(exc).__name__,
            "path": request.url.path,
            "traceback": traceback.format_exc()
        }
    )

MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def call_gemini(system_prompt: str, contents: list, temperature: float = 0.2, response_mime_type: Optional[str] = None) -> str:
    """Direct HTTP call to Gemini API - zero external dependencies, ultra-fast."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set in Vercel environment variables. Please add it in Vercel Project Settings -> Environment Variables."
        )
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={api_key}"
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature
        }
    }
    
    if system_prompt:
        payload["system_instruction"] = {
            "parts": [{"text": system_prompt}]
        }
        
    if response_mime_type:
        payload["generationConfig"]["responseMimeType"] = response_mime_type
        
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            if not candidates:
                raise HTTPException(status_code=500, detail="Gemini returned no response candidates.")
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise HTTPException(status_code=500, detail="Gemini response content is empty.")
            return parts[0].get("text", "")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        try:
            err_json = json.loads(error_body)
            err_msg = err_json.get("error", {}).get("message", error_body)
        except Exception:
            err_msg = error_body
        raise HTTPException(status_code=e.code, detail=f"Gemini API error ({e.code}): {err_msg}")
    except urllib.error.URLError as e:
        raise HTTPException(status_code=500, detail=f"Network error calling Gemini: {str(e.reason)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# ==========================================
# Health Check & Root Endpoints (all path variations)
# ==========================================

async def _health_response():
    has_key = bool(os.getenv("GEMINI_API_KEY"))
    return {
        "status": "online",
        "service": "MediKiosk Clinical Intake & Triage API",
        "version": "1.0.0",
        "gemini_api_key_configured": has_key,
        "model": MODEL_ID,
        "endpoints": [
            "/triage/emergency-check",
            "/kiosk/chat",
            "/ayush/chat",
            "/docs"
        ]
    }

@app.get("/")
@app.get("/api")
@app.get("/api/")
@app.get("/api/index")
@app.get("/api/index.py")
@app.get("/api/health")
async def health_check():
    return await _health_response()


# ==========================================
# 1. EMERGENCY DETECTION CLASSIFIER
# ==========================================

EMERGENCY_SYSTEM_PROMPT = """You are an emergency-detection classifier for a hospital patient-intake system. Given a short piece of patient-reported text, decide if it represents a possible medical emergency requiring immediate triage.

Respond ONLY in this JSON format:
{"emergency": true/false, "confidence": "high/medium/low", "reason": "<one short phrase>"}

Treat as emergencies: chest pain with sweating/breathlessness, sudden one-sided weakness or facial drooping or slurred speech, severe uncontrolled bleeding, difficulty breathing at rest, loss of consciousness, severe allergic reaction (facial/throat swelling), seizure, suicidal ideation, vomiting blood.

Do NOT flag routine symptoms (mild fever, cough, headache, mild back pain, common cold).
"""

class EmergencyRequest(BaseModel):
    patient_text: str

class EmergencyResponse(BaseModel):
    emergency: bool
    confidence: str
    reason: str

async def _process_emergency(req: EmergencyRequest):
    contents = [
        {"role": "user", "parts": [{"text": f"Patient text: {req.patient_text}"}]}
    ]
    raw_text = call_gemini(
        system_prompt=EMERGENCY_SYSTEM_PROMPT,
        contents=contents,
        temperature=0.0,
        response_mime_type="application/json"
    )
    try:
        return json.loads(raw_text)
    except Exception:
        return {"emergency": False, "confidence": "low", "reason": raw_text}

@app.post("/triage/emergency-check", response_model=EmergencyResponse)
@app.post("/api/triage/emergency-check", response_model=EmergencyResponse)
@app.post("/api/index.py/triage/emergency-check", response_model=EmergencyResponse)
async def check_emergency(req: EmergencyRequest):
    return await _process_emergency(req)


# ==========================================
# 2. MEDIKIOSK (SOCRATES CLINICAL HISTORY)
# ==========================================

MEDIKIOSK_SYSTEM_PROMPT = """You are a clinical history-taking assistant for an Indian hospital kiosk called MediKiosk. A patient has stated their chief complaint. Ask ONE focused follow-up question at a time, following the SOCRATES framework (Site, Onset, Character, Radiation, Associations, Time course, Exacerbating/relieving factors, Severity), until you have enough for a complete History of Present Illness.

Rules:
- Ask only ONE question at a time, in simple, non-technical language.
- Never repeat ground the patient has already covered.
- After 6-8 exchanges (fewer for minor complaints), stop and output ONLY JSON with keys: 
  chiefComplaint, onset, duration, character, location, radiation, aggravatingFactors, relievingFactors, severity, associatedSymptoms.
- If the patient mentions an emergency pattern, stop immediately and output:
  {"emergency": true, "reason": "<short reason>"}
- Never diagnose. Never suggest medication. You are only collecting history.
- Match the patient's language if they respond in Hindi/Tamil/Telugu/Kannada/Bengali.
"""

class Message(BaseModel):
    role: str
    text: str

class ChatSessionRequest(BaseModel):
    chief_complaint: Optional[str] = None
    messages: List[Message] = []

async def _process_kiosk_chat(req: ChatSessionRequest):
    contents = []
    if req.chief_complaint and not req.messages:
        contents.append({
            "role": "user", 
            "parts": [{"text": f"The patient's chief complaint is: {req.chief_complaint}"}]
        })
    else:
        for m in req.messages:
            contents.append({
                "role": "user" if m.role == "user" else "model",
                "parts": [{"text": m.text}]
            })

    response_text = call_gemini(
        system_prompt=MEDIKIOSK_SYSTEM_PROMPT,
        contents=contents,
        temperature=0.2
    ).strip()

    if response_text.startswith("{") and response_text.endswith("}"):
        try:
            return {"type": "completed", "data": json.loads(response_text)}
        except Exception:
            pass
    
    return {"type": "question", "reply": response_text}

@app.post("/kiosk/chat")
@app.post("/api/kiosk/chat")
@app.post("/api/index.py/kiosk/chat")
async def medikiosk_chat(req: ChatSessionRequest):
    return await _process_kiosk_chat(req)


# ==========================================
# 3. AYUSH DASHAVIDHA PARIKSHA ASSISTANT
# ==========================================

AYUSH_SYSTEM_PROMPT = """You are an Ayurvedic clinical history assistant conducting a Dashavidha Pariksha assessment for a patient at an AYUSH OPD. Guide them through a friendly, plain-language conversation (not a form) to gather these ten parameters, one at a time - NEVER use the Sanskrit terms with the patient, translate the intent instead:

1. Prakriti - typical body build, skin, hair, appetite, temperament since childhood
2. Vikriti - what feels different from their usual self recently
3. Sara - general energy and vitality
4. Samhanana - physical build/frame
5. Pramana - height/build, kept simple
6. Satmya - foods, weather, or habits that suit them best
7. Sattva - how they handle stress or emotional situations
8. Ahara Shakti - appetite, digestion, food preferences
9. Vyayama Shakti - how much physical activity they can comfortably do
10. Vaya - age and general vitality for their age group

Rules:
- One simple, conversational question at a time; keep the whole flow under 10 questions.
- At the end, output ONLY JSON with keys: prakriti, vikriti, sara, samhanana, pramana, satmya, sattva, aharaShakti, vyayamaShakti, vaya.
- Never diagnose a dosha imbalance - just record what the patient reports for the physician to interpret.
"""

async def _process_ayush_chat(req: ChatSessionRequest):
    contents = []
    if not req.messages:
        contents.append({
            "role": "user", 
            "parts": [{"text": "Start the assessment with the first friendly question."}]
        })
    else:
        for m in req.messages:
            contents.append({
                "role": "user" if m.role == "user" else "model",
                "parts": [{"text": m.text}]
            })

    response_text = call_gemini(
        system_prompt=AYUSH_SYSTEM_PROMPT,
        contents=contents,
        temperature=0.3
    ).strip()

    if response_text.startswith("{") and response_text.endswith("}"):
        try:
            return {"type": "completed", "data": json.loads(response_text)}
        except Exception:
            pass

    return {"type": "question", "reply": response_text}

@app.post("/ayush/chat")
@app.post("/api/ayush/chat")
@app.post("/api/index.py/ayush/chat")
async def ayush_chat(req: ChatSessionRequest):
    return await _process_ayush_chat(req)