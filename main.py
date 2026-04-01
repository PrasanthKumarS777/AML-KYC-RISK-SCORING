from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.utils.config import APP_NAME, APP_VERSION
from src.core.database import init_db

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="AML/KYC Customer Risk Re-Scoring System — Compliance Analytics API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

init_db()

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Health"])
def root():
    return {
        "app":     APP_NAME,
        "version": APP_VERSION,
        "status":  "running",
        "docs":    "/docs"
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}