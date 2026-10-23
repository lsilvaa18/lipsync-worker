# Worker de lip sync (LatentSync 1.6, Apache-2.0) para RunPod Serverless.
# GPU: 24 GB (L4 / A5000 / 3090) basta; o 1.6 usa ~18 GB.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip git ffmpeg libgl1 libglib2.0-0 \
      build-essential curl unzip ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Código do LatentSync numa versão fixa (não muda sem querermos).
ARG LATENTSYNC_COMMIT=a229c3948406bc2cf6eaf4873e662e70c6a04746
RUN git clone https://github.com/bytedance/LatentSync /app \
    && cd /app && git checkout ${LATENTSYNC_COMMIT}
WORKDIR /app

# gradio só serve para a interface web deles; fora para a imagem ficar menor.
RUN grep -v '^gradio' requirements.txt > /tmp/req.txt \
    && python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/req.txt \
    && python -m pip install 'runpod>=1.7,<2' requests==2.32.3

# Pesos dentro da imagem: sem download no cold start.
RUN huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints \
    && huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints \
    && huggingface-cli download stabilityai/sd-vae-ft-mse \
    && mkdir -p checkpoints/auxiliary/models \
    && curl -L -o /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
    && mkdir -p /tmp/bl checkpoints/auxiliary/models/buffalo_l \
    && unzip -q /tmp/buffalo_l.zip -d /tmp/bl \
    && find /tmp/bl -name '*.onnx' -exec mv {} checkpoints/auxiliary/models/buffalo_l/ \; \
    && rm -rf /tmp/bl /tmp/buffalo_l.zip

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
