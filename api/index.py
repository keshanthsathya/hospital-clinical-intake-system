import os
import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# Verify that an API key is available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    pass

app = FastAPI(title="Clinical Intake & Triage System", version="1.0.0")

# CORS middleware
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GenAI Client
client = genai.Client()
MODEL_ID = "gemini-2.5-flash"


# ==========================================
# Health check
# ==========================================

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "Clinical Intake & Triage System", "version": "1.0.0"}


@app.get("/api")
async def api_health():
    return {"status": "ok", "service": "Clinical Intake & Triage System", "version": "1.0.0"}


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

@app.post("/api/triage/emergency-check", response_model=EmergencyResponse)
async def check_emergency(req: EmergencyRequest):
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                {"role": "user", "parts": [{"text": f"Patient text: {req.patient_text}"}]}
            ],
            config=types.GenerateContentConfig(
                system_instruction=EMERGENCY_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.0
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

@app.post("/api/kiosk/chat")
async def medikiosk_chat(req: ChatSessionRequest):
    try:
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

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=MEDIKIOSK_SYSTEM_PROMPT,
                temperature=0.2
            ),
        )

        response_text = response.text.strip()
        if response_text.startswith("{") and response_text.endswith("}"):
            return {"type": "completed", "data": json.loads(response_text)}
        
        return {"type": "question", "reply": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

@app.post("/api/ayush/chat")
async def ayush_chat(req: ChatSessionRequest):
    try:
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

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=AYUSH_SYSTEM_PROMPT,
                temperature=0.3
            ),
        )

        response_text = response.text.strip()
        if response_text.startswith("{") and response_text.endswith("}"):
            return {"type": "completed", "data": json.loads(response_text)}

        return {"type": "question", "reply": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))