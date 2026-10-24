# Worker de upscale (Real-ESRGAN x4plus, BSD-3) para RunPod Serverless.
# GPU: 24 GB (L4 / A5000 / 3090) sobra.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install 'spandrel==0.4.1' 'runpod>=1.7,<2' requests==2.32.3 numpy

# Pesos dentro da imagem: sem download no cold start.
RUN mkdir -p /models && curl -fL -o /models/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "handler.py"]
