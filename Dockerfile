FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu129
ARG INSTALL_FLASHINFER=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/home/app/.cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/home/app/.cache/inductor \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        libsndfile1 \
        ninja-build \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --no-cache-dir \
        --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" \
        "torchaudio==${TORCH_VERSION}"

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /home/app/.cache/huggingface /home/app/.cache/inductor /app/models /app/voices \
    && chown -R app:app /home/app /app/models /app/voices

WORKDIR /app

COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app omnivoice_server.py omnivoice.py omnivoice_flashinfer.py inference.py ./
COPY --chown=app:app web ./web

RUN /opt/venv/bin/pip install --no-cache-dir \
        --extra-index-url "${TORCH_INDEX}" . \
    && if [ "${INSTALL_FLASHINFER}" = "true" ]; then \
        /opt/venv/bin/pip install --no-cache-dir flashinfer-python==0.6.15.post1; \
        /opt/venv/bin/pip install --no-cache-dir --no-deps \
            "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.15.post1/flashinfer_jit_cache-0.6.15.post1+cu129-cp39-abi3-manylinux_2_28_x86_64.whl"; \
    fi

USER app

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8003/health >/dev/null || exit 1

STOPSIGNAL SIGINT

ENTRYPOINT ["/opt/venv/bin/python", "omnivoice_server.py"]
CMD ["--host", "0.0.0.0", "--port", "8003"]
