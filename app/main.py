"""
Bhairava — Fraud Detection System
app/main.py

FastAPI application entry point for Bhairava AI.
Provides sub-50ms REST endpoints for real-time transaction risk scoring
and automated merchant defense actions.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import router as api_v1_router


app = FastAPI(
    title="Bhairava AI — Fraud Detector & Auto-Responder",
    description=(
        "Production-grade, two-stage fraud prevention system for Razorpay merchants. "
        "Stage 1 calculates machine learning risk scores in real-time. "
        "Stage 2 automatically executes defensive actions (Allow, Step-Up 3DS Challenge, Auto-Decline) "
        "with complete explainability and audit trails."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend simulators and merchant dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(api_v1_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "system": "Bhairava AI",
        "tagline": "Real-Time Fraud Detection & Automated Response Engine",
        "buildathon": "Razorpay AI Buildathon 2026",
        "track": "Track 2: AI Risk Manager",
        "docs": "/docs",
        "endpoints": {
            "predict_fraud": "/api/v1/predict-fraud",
            "auto_respond": "/api/v1/auto-respond",
            "health": "/api/v1/health",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
