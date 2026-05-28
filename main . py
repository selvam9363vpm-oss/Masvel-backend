import os
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, validator
from dotenv import load_dotenv
import time

load_dotenv()

app = FastAPI(
    title="Masvel AI Image Generation API",
    description="Backend for Masvel AI — SDXL image generation via Hugging Face",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Update ALLOWED_ORIGINS with your real Netlify URL after deploy
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

# Simple in-memory rate limiting (per IP, resets every 60s)
rate_limit_store: dict = {}
RATE_LIMIT = 5        # max requests
RATE_WINDOW = 60      # seconds

# ── MODELS ────────────────────────────────────────────────────────────────────
class PromptRequest(BaseModel):
    prompt: str

    @validator("prompt")
    def validate_prompt(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Prompt cannot be empty.")
        if len(v) > 500:
            raise ValueError("Prompt must be 500 characters or fewer.")
        return v

# ── HELPERS ───────────────────────────────────────────────────────────────────
def check_rate_limit(ip: str) -> bool:
    """Returns True if request is allowed, False if rate limited."""
    now = time.time()
    if ip not in rate_limit_store:
        rate_limit_store[ip] = []
    # Purge old entries
    rate_limit_store[ip] = [t for t in rate_limit_store[ip] if now - t < RATE_WINDOW]
    if len(rate_limit_store[ip]) >= RATE_LIMIT:
        return False
    rate_limit_store[ip].append(now)
    return True

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "Masvel AI backend is running ✨"}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "stabilityai/stable-diffusion-xl-base-1.0"}

@app.post("/generate-masvel-image")
async def generate_image(request: PromptRequest, req: Request):
    # Guard: token must be present
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Server misconfiguration: API key missing.")

    # Rate limiting
    client_ip = req.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment before generating again."
        )

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": request.prompt,
        "parameters": {
            "num_inference_steps": 30,   # good quality/speed balance for mobile
            "guidance_scale": 7.5,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            hf_response = await client.post(API_URL, headers=headers, json=payload)
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Image generation timed out. The model may be loading — please try again in 30 seconds."
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Network error contacting Hugging Face: {str(e)}")

    if hf_response.status_code == 200:
        return Response(
            content=hf_response.content,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "X-Generated-By": "Masvel-AI"
            }
        )
    elif hf_response.status_code == 503:
        # Model is loading (cold start)
        raise HTTPException(
            status_code=503,
            detail="Model is warming up. Please retry in 20–40 seconds."
        )
    elif hf_response.status_code == 401:
        raise HTTPException(status_code=401, detail="Invalid or expired API key on server.")
    else:
        raise HTTPException(
            status_code=hf_response.status_code,
            detail=f"Hugging Face error: {hf_response.text[:200]}"
        )