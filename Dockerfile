# Imagem enxuta para a VPS (1 vCPU, 2 GB RAM): sem PyTorch, só ONNX Runtime.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRONTEND_DIR=/app/frontend

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/app backend/app
COPY frontend frontend
COPY models models
# Garante leitura dos modelos mesmo se o arquivo foi copiado com permissão restrita.
RUN chmod -R a+rX /app/models

RUN useradd --create-home --uid 1000 app
USER app

WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

# Um único worker: a análise é limitada pela CPU e o modelo fica uma vez na memória.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
