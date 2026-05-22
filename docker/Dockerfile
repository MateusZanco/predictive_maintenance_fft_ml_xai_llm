FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY app/frontend/package.json ./
RUN npm install
COPY app/frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_PORT=8000
ENV LLM_CHAT_COMPLETIONS_URL=http://127.0.0.1:8080/v1/chat/completions
ENV LLM_MODEL=
ENV LLM_TEMPERATURE=0.2
ENV LLM_TOP_P=0.8
ENV LLM_MAX_TOKENS=500
ENV LLM_TIMEOUT_SECONDS=250

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY app/backend/requirements.txt /tmp/backend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/backend-requirements.txt

COPY app/backend /app/app/backend
COPY --from=frontend-builder /frontend/dist /app/app/backend/static

ENV ROCKPI_SAMPLE_DIR=/app/runtime_samples
ENV ROCKPI_MODEL_ARTIFACTS_DIR=/app/model_artifacts
ENV ROCKPI_STATIC_DIR=/app/app/backend/static

CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${APP_PORT:-8000}"]
