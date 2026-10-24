"""
RunPod Serverless handler: upscale de vídeo com Real-ESRGAN x4plus (vídeo -> 1080p).

Entrada (job["input"]):
  video_url | video_base64   vídeo de origem (ex.: 832x464)
  start                      segundo inicial do trecho (padrão 0)
  duration                   duração do trecho em segundos (padrão: até o fim)
  width, height              tamanho final (padrão 1920x1080)
  blend                      0 a 1: parte do upscale "liso" (lanczos) misturada para
                             manter textura de pele (padrão 0.2)
  crf                        qualidade do x264 (padrão 16)

Saída: {"video_base64": "...", "bytes": n, "frames": n, "seconds": s, "inference_s": s}
O trecho sai SEM áudio; o app costura os trechos e coloca a voz.
Mantenha trechos de ~10 s: o resultado volta pela resposta do RunPod (limite de poucos MB).
"""

import base64
import os
import shutil
import subprocess
import time
import uuid

import numpy as np
import requests
import runpod
import torch
import torch.nn.functional as F
from spandrel import ModelLoader

MAX_DOWNLOAD = 500 * 1024 * 1024

t0 = time.time()
MODEL = ModelLoader().load_from_file("/models/RealESRGAN_x4plus.pth").cuda().eval().half()
print(f"modelo carregado em {time.time() - t0:.1f}s", flush=True)


def fetch(inp: dict, dest: str) -> None:
    url, b64 = inp.get("video_url"), inp.get("video_base64")
    if b64:
        with open(dest, "wb") as f:
            f.write(base64.b64decode(b64))
        return
    if not url:
        raise ValueError("faltou video_url ou video_base64")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise ValueError("vídeo maior que 500 MB")
                f.write(chunk)


def probe(path: str):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    w, h, fr = int(out[0]), int(out[1]), out[2]
    num, den = fr.split("/")
    return w, h, float(num) / float(den)


@torch.inference_mode()
def enhance(frame: np.ndarray, out_w: int, out_h: int, blend: float) -> np.ndarray:
    x = torch.from_numpy(frame).cuda().permute(2, 0, 1).unsqueeze(0).half() / 255.0
    y = MODEL(x)  # 4x
    y = F.interpolate(y.float(), size=(out_h, out_w), mode="area")
    if blend > 0:
        base = F.interpolate(x.float(), size=(out_h, out_w), mode="bicubic", align_corners=False)
        y = y * (1 - blend) + base * blend
    y = (y.clamp(0, 1) * 255.0).round().byte().squeeze(0).permute(1, 2, 0)
    return y.cpu().numpy()


def handler(job):
    inp = job["input"]
    work = f"/tmp/job-{uuid.uuid4().hex}"
    os.makedirs(work)
    try:
        src, out = f"{work}/in.mp4", f"{work}/out.mp4"
        runpod.serverless.progress_update(job, "baixando vídeo")
        fetch(inp, src)

        w, h, fps = probe(src)
        out_w, out_h = int(inp.get("width", 1920)), int(inp.get("height", 1080))
        blend = float(min(1.0, max(0.0, inp.get("blend", 0.2))))
        crf = str(int(inp.get("crf", 16)))
        start = float(inp.get("start", 0))
        cut = ["-ss", str(start)]
        if inp.get("duration"):
            cut += ["-t", str(float(inp["duration"]))]

        dec = subprocess.Popen(
            ["ffmpeg", "-v", "error", *cut, "-i", src, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE,
        )
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-crf", crf, "-preset", "fast", "-pix_fmt", "yuv420p", out],
            stdin=subprocess.PIPE,
        )
        size = w * h * 3
        n = 0
        runpod.serverless.progress_update(job, "melhorando a imagem")
        t = time.time()
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            enc.stdin.write(enhance(frame, out_w, out_h, blend).tobytes())
            n += 1
        enc.stdin.close()
        dec.wait()
        if enc.wait() != 0 or n == 0:
            raise RuntimeError(f"ffmpeg falhou ({n} quadros)")
        took = time.time() - t
        print(f"{n} quadros em {took:.1f}s ({n / took:.1f} q/s)", flush=True)

        with open(out, "rb") as f:
            data = f.read()
        return {"video_base64": base64.b64encode(data).decode(), "bytes": len(data),
                "frames": n, "seconds": n / fps, "inference_s": took}
    except Exception as e:  # noqa: BLE001 - devolve a mensagem para o app mostrar
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
