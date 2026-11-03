FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY sops ./sops
COPY web ./web

# data/ holds the audit log and session checkpoints; mount a volume there to keep them.
RUN useradd --create-home bot && mkdir data && chown bot data
USER bot

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"

# The rate limiter keys on the last X-Forwarded-For hop, so it works behind one proxy.
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port $PORT"]
