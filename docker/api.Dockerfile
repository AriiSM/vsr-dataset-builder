# vsr-api — creierul de comandă: FastAPI peste dataset.db. FĂRĂ torch, FĂRĂ GPU.
# Build (din rădăcina repo-ului):
#   docker build -f docker/api.Dockerfile -t vsr-api --build-arg GIT_SHA=$(git rev-parse --short HEAD) .

FROM python:3.11-slim

# Layerele de dependențe ÎNAINTEA codului → schimbarea codului nu reinstalează pip
COPY backend/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY backend/api/          backend/api/
COPY backend/shared/       backend/shared/
COPY backend/run_api.py    backend/run_api.py

# Volumele montează peste acestea la runtime (compose)
RUN mkdir -p data config && \
    useradd --create-home vsr && chown -R vsr:vsr /app
USER vsr

ARG GIT_SHA=""
ENV GIT_SHA=${GIT_SHA} \
    VSR_CONFIG=/app/config/config.yaml \
    VSR_CATALOG_DIR=/app/data/catalog \
    VSR_PROCESSED_DIR=/app/data/processed \
    VSR_LOGS_DIR=/app/data/logs \
    VSR_CACHE_DIR=/app/data/cache \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["python", "backend/run_api.py", "--host", "0.0.0.0", "--port", "8000"]
