# FastAPI service. The model is not in git: it is downloaded from the GitHub
# release at build time (override MODEL_URL to serve your own trained artifact).
FROM python:3.11-slim

ARG MODEL_URL=https://github.com/Nardy11/transaction-classifier-mlops/releases/download/v1.0.0/model.joblib
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TXCLF_MODEL=/app/models/model.joblib

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
ADD ${MODEL_URL} /app/models/model.joblib

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "txclf.api:app", "--host", "0.0.0.0", "--port", "8000"]
