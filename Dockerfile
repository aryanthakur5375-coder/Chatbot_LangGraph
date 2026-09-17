FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss / torch wheels at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persisted SQLite checkpoint DB lives here; mount a volume in production
ENV CHATBOT_DB_PATH=/app/data/chatbot.db
RUN mkdir -p /app/data

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# No secrets are baked into the image -- OPENROUTER_API_KEY must be
# supplied at runtime via `docker run -e` or an env file.
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
