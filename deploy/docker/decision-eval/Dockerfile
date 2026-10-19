FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10 PYTHONUNBUFFERED=1
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install pip==25.2 --index-url https://pypi.org/simple/
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
ARG TORCHVISION_PACKAGE=torchvision==0.24.1+cu126
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --no-deps "${TORCHVISION_PACKAGE}" --index-url https://download.pytorch.org/whl/cu126
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --index-url https://pypi.org/simple/ \
    transformers==5.17.0 accelerate==1.12.0 fastapi==0.135.1 uvicorn==0.41.0 \
    pillow==12.1.1 safetensors==0.8.0
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --no-deps \
    'laya @ https://files.pythonhosted.org/packages/b8/f6/9e3f039d08319e153721210667548e2cac1679b56461c8b5e81acc93b4db/laya-0.4.1-py3-none-any.whl#sha256=5252730d8be604aecdc6c52adeb591e75d2b97ef3f52278cba9b37e403c784c7'
RUN python -m pip check

WORKDIR /app
COPY server.py /app/server.py
COPY decision_reranker.py /app/decision_reranker.py
USER 10001:10001
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--limit-concurrency", "8"]
