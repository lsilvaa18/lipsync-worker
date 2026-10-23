"""
RunPod Serverless handler: LatentSync 1.6 (vídeo + áudio -> vídeo com a boca sincronizada).

Entrada (job["input"]):
  video_url | video_base64     vídeo de referência (rosto de frente)
  audio_url | audio_base64     a fala nova
  output_upload_url            opcional: URL de upload (PUT) para o resultado
  inference_steps              20 a 50, padrão 20 (mais = melhor e mais lento)
  guidance_scale               1.0 a 3.0, padrão 1.5 (mais = boca mais fiel, pode tremer)
  seed                         padrão 1247

Saída:
  com output_upload_url: {"uploaded": true, "bytes": n, "seconds": s}
  sem:                   {"video_base64": "...", "bytes": n, "seconds": s}  (só para clipes curtos)

O modelo carrega uma vez por worker, fora do handler.
"""

import base64
import os
import shutil
import subprocess
import time
import uuid

import requests
import runpod
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed
from DeepCache import DeepCacheSDHelper

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature

CONFIG = OmegaConf.load("configs/unet/stage2_512.yaml")
CKPT = "checkpoints/latentsync_unet.pt"
MAX_DOWNLOAD = 500 * 1024 * 1024


def load_pipeline():
    fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if fp16 else torch.float32
    scheduler = DDIMScheduler.from_pretrained("configs")
    audio_encoder = Audio2Feature(
        model_path="checkpoints/whisper/tiny.pt",
        device="cuda",
        num_frames=CONFIG.data.num_frames,
        audio_feat_length=CONFIG.data.audio_feat_length,
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(CONFIG.model), CKPT, device="cpu")
    unet = unet.to(dtype=dtype)
    pipe = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler).to("cuda")
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()
    return pipe, dtype


t0 = time.time()
PIPE, DTYPE = load_pipeline()
print(f"modelo carregado em {time.time() - t0:.1f}s", flush=True)


def fetch(inp: dict, key: str, dest: str) -> None:
    url, b64 = inp.get(f"{key}_url"), inp.get(f"{key}_base64")
    if b64:
        with open(dest, "wb") as f:
            f.write(base64.b64decode(b64))
        return
    if not url:
        raise ValueError(f"faltou {key}_url ou {key}_base64")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise ValueError(f"{key} maior que 500 MB")
                f.write(chunk)


def duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip() or 0)


def handler(job):
    inp = job["input"]
    work = f"/tmp/job-{uuid.uuid4().hex}"
    os.makedirs(work)
    try:
        video, audio, out = f"{work}/in.mp4", f"{work}/in_audio", f"{work}/out.mp4"
        runpod.serverless.progress_update(job, "baixando arquivos")
        fetch(inp, "video", video)
        fetch(inp, "audio", audio)

        # O LatentSync quer 25 fps; e o áudio em wav evita surpresa de codec.
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-r", "25", "-an", "-c:v", "libx264",
                        "-crf", "18", f"{work}/v25.mp4"], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", audio, "-ar", "16000", "-ac", "1",
                        f"{work}/a.wav"], check=True)

        steps = int(min(50, max(20, inp.get("inference_steps", 20))))
        guidance = float(min(3.0, max(1.0, inp.get("guidance_scale", 1.5))))
        set_seed(int(inp.get("seed", 1247)))

        runpod.serverless.progress_update(job, "sincronizando a boca")
        t = time.time()
        PIPE(
            video_path=f"{work}/v25.mp4",
            audio_path=f"{work}/a.wav",
            video_out_path=out,
            num_frames=CONFIG.data.num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            weight_dtype=DTYPE,
            width=CONFIG.data.resolution,
            height=CONFIG.data.resolution,
            mask_image_path=CONFIG.data.mask_image_path,
            temp_dir=f"{work}/temp",
        )
        took = time.time() - t
        secs = duration(out)
        print(f"{secs:.1f}s de vídeo em {took:.1f}s", flush=True)

        upload = inp.get("output_upload_url")
        if upload:
            with open(out, "rb") as f:
                r = requests.put(upload, data=f, headers={"Content-Type": "video/mp4"}, timeout=300)
            r.raise_for_status()
            return {"uploaded": True, "bytes": os.path.getsize(out), "seconds": secs, "inference_s": took}

        # Sem URL de upload: devolve comprimido em base64 (clipes curtos de teste).
        small = f"{work}/out_small.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", out, "-c:v", "libx264", "-crf", "24",
                        "-preset", "medium", "-c:a", "aac", "-b:a", "128k", small], check=True)
        with open(small, "rb") as f:
            data = f.read()
        return {"video_base64": base64.b64encode(data).decode(), "bytes": len(data),
                "seconds": secs, "inference_s": took}
    except Exception as e:  # noqa: BLE001 - devolve a mensagem para o app mostrar
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
