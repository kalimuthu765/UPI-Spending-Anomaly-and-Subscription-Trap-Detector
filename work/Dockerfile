# Multi-stage build for UPI Spending Anomaly & Subscription Trap Detector (Unified Single Container)

# Stage 1: Build React Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Final Backend + ML Pipeline
FROM python:3.11-slim

WORKDIR /app

# System dependencies: poppler for PDF parsing, libpq-dev for Postgres
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code & ML models
COPY ml/ ./ml/
COPY backend/ ./backend/

# Copy compiled React frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Writable directories for models and runtime data
RUN mkdir -p /app/data/processed /app/data/synthetic /app/ml/artifacts/user_models
VOLUME ["/app/ml/artifacts/user_models"]

ENV PORT=80
EXPOSE 80
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-80}"]
