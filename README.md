# MediKiosk – Clinical Intake & Triage API

> FastAPI backend powering the MediKiosk hospital kiosk ecosystem with Gemini AI.

![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-AI-4285F4?logo=google&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/triage/emergency-check` | Emergency detection classifier |
| `POST` | `/kiosk/chat` | SOCRATES clinical history chatbot |
| `POST` | `/ayush/chat` | AYUSH Dashavidha Pariksha assistant |

---

## Local Development

```bash
# Clone
git clone https://github.com/keshanthsathya/hospital-clinical-intake-system.git
cd hospital-clinical-intake-system

# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add your real GEMINI_API_KEY

# Run
uvicorn app:app --reload --port 8000
```

API docs at **http://localhost:8000/docs**

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml`
5. Add environment variable: `GEMINI_API_KEY` = your key
6. Add environment variable: `ALLOWED_ORIGINS` = your Vercel frontend URLs (comma-separated)
7. Deploy!

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key from [AI Studio](https://aistudio.google.com/apikey) |
| `ALLOWED_ORIGINS` | No | Comma-separated list of allowed CORS origins (defaults to `*`) |

---

## Connected Frontends

- [medikiosk-frontend](https://github.com/keshanthsathya/medikiosk-frontend) – Patient kiosk UI (React + Vite)
- [Medikiosk-Doctor](https://github.com/keshanthsathya/Medikiosk-Doctor) – Doctor dashboard UI

---

> Built for **Smart India Hackathon (SIH)**