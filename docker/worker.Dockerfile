# vsr-worker — mușchiul: consumatorul cozii, rulează pipeline-ul pe GPU.
# Aceeași imagine servește și review_assistant (comandă diferită în compose).
#
# Build (din rădăcina repo-ului; cere TalkNet-ASD/ clonat acolo):
#   docker build -f docker/worker.Dockerfile -t vsr-worker --build-arg GIT_SHA=$(git rev-parse --short HEAD) .
#
# Baza CUDA: torch cu128 își aduce bibliotecile CUDA în wheel, dar cudnn-runtime
# acoperă onnxruntime-gpu + orice fallback; versiunea EXACTĂ se confirmă la
# pilot pe mașina reală (nvidia-smi → driver → matricea de compatibilitate).

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip ffmpeg git \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python

# 1. torch cu CUDA (layerul cel mai greu — primul, se schimbă cel mai rar)
RUN python -m pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. restul stack-ului ML (torch>= din listă e deja satisfăcut)
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -m pip install --no-cache-dir onnxruntime-gpu

# 3. TalkNet vendorizat (clonat în rădăcina repo-ului înainte de build).
# Repo-ul oficial NU are setup.py — nu se instalează cu pip; codul îl
# importă direct (`from talkNet import ...`), deci ajunge pe PYTHONPATH.
WORKDIR /app
COPY TalkNet-ASD/ TalkNet-ASD/
ENV PYTHONPATH=/app/TalkNet-ASD

# 4. codul (ultimul layer — se schimbă cel mai des)
COPY backend/  backend/
COPY config/   config/

# Volumele compose montează peste: /app/data /app/models /app/config
RUN mkdir -p data models

ARG GIT_SHA=""
ENV GIT_SHA=${GIT_SHA}

# Cache-urile de modele sunt redirecționate sub /app/models de
# vsr_shared/model_env.py (TORCH_HOME/HF_HOME/INSIGHTFACE_HOME) — worker-ul
# o face singur la pornire; nimic nu se descarcă în imagine.

CMD ["python", "backend/run_worker.py"]
