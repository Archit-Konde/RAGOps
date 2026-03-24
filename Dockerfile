FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for layer caching (locked versions)
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Copy application source
COPY . .

# Run as non-root user
RUN adduser --disabled-password --no-create-home appuser
USER appuser

EXPOSE 7860

CMD ["uvicorn", "apps.api.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
