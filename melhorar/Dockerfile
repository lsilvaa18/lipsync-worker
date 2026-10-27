# Worker "melhorar" para RunPod Serverless: fundo 1080p do frame original + rosto restaurado (GFPGAN v1.4, Apache-2.0).
# GPU: 24 GB (L4 / A5000 / 3090) sobra.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install 'spandrel==0.4.1' 'opencv-python-headless<5' 'runpod>=1.7,<2' numpy

# Pesos dentro da imagem: sem download no cold start.
RUN mkdir -p /models && curl -fL -o /models/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "handler.py"]
