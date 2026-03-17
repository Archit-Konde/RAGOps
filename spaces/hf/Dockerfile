FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (layer cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Pre-download ML models into the image to avoid cold-start latency
ENV HF_HOME=/app/.hf_cache
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models cached successfully')"

# HF Spaces requires non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "apps.api.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
