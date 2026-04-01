# ── Base Image ──────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ────────────────────────────────────────────────
LABEL maintainer="Prasanth Kumar Sahu"
LABEL project="AML-KYC-Risk-Scoring"
LABEL version="1.0.0"

# ── Environment Variables ────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── Working Directory ────────────────────────────────────────
WORKDIR /app

# ── System Dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python Dependencies ──────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy Project Files ───────────────────────────────────────
COPY . .
RUN pip install -e .

# ── Create necessary directories ─────────────────────────────
RUN mkdir -p data/raw data/processed data/synthetic \
             src/models/saved logs

# ── Expose FastAPI Port ──────────────────────────────────────
EXPOSE 8000

# ── Startup: Generate data + train + load DB + start API ─────
CMD ["sh", "-c", "\
    python -m src.core.data_generator && \
    python -m src.features.risk_features && \
    python -m src.models.train_model && \
    python -m src.models.explainer && \
    python -m src.core.database && \
    python -m src.pipelines.load_to_db && \
    uvicorn main:app --host 0.0.0.0 --port 8000"]