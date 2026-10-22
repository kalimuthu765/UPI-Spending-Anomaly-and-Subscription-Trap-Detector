# Backend — FastAPI + ML pipeline
FROM python:3.11-slim

WORKDIR /app

# System deps: pdfplumber needs poppler for some PDF operations,
# psycopg2 needs libpq for Postgres
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY ml/ ./ml/
COPY backend/ ./backend/

# Data folders for the legacy demo endpoints. They are created empty here (the
# generated CSVs are not part of the source download); regenerate them with the
# commands in README "Quick start" if you want those endpoints. The main app
# (upload -> personal model -> dashboard) does not need them.
RUN mkdir -p /app/data/processed /app/data/synthetic

# Writable dirs created at runtime (SQLite DB if no DATABASE_URL given,
# per-user model artifacts) — declared so a volume can be mounted here
RUN mkdir -p /app/ml/artifacts/user_models
VOLUME ["/app/ml/artifacts/user_models"]

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
