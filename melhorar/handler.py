"""
RunPod Serverless handler: melhora um trecho de talking head (InfiniteTalk 832x464 -> 1080p).
  1) escala o trecho para 1920x1080;
  2) fora da faixa central (onde a pessoa se mexe) usa o frame original em 1080p (fundo nítido);
  3) restaura o rosto com GFPGAN v1.4.

Entrada (job["input"]):
  video_base64   trecho do talking.mp4 (~10 s, sem áudio)
  still_base64   frame original 1920x1080 (still.jpg)
  face_weight    0 a 1, força da restauração (padrão 0.7)
  crf            qualidade do x264 (padrão 16)

Saída: {"video_base64": "...", "bytes": n, "frames": n, "inference_s": s}
O trecho sai SEM áudio; o app costura os trechos e coloca a voz.
"""

import base64
import os
import shutil
import subprocess
import time
import uuid

import cv2
import numpy as np
import runpod
import torch
from spandrel import ModelLoader

W, H = 1920, 1080

t0 = time.time()
MODEL = ModelLoader().load_from_file("/models/GFPGANv1.4.pth").cuda().eval()
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print(f"modelo carregado em {time.time() - t0:.1f}s", flush=True)

# Máscara do fundo: faixa central de cima a baixo (cabelo, mãos e cadeira se mexem), borda suave.
BG_MASK = np.zeros((H, W), np.float32)
BG_MASK[:, 440:1500] = 1
BG_MASK = cv2.GaussianBlur(BG_MASK, (0, 0), 35)[..., None]

YY, XX = np.mgrid[0:512, 0:512]
FACE_MASK = (np.clip(1.2 - np.hypot((XX - 256) / 200, (YY - 270) / 235), 0, 1) ** 0.8)[..., None].astype(np.float32)


def probe_fps(path: str) -> float:
    fr = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
         "-of", "csv=p=0", path], capture_output=True, text=True, check=True,
    ).stdout.strip()
    num, den = fr.split("/")
    return float(num) / float(den)


@torch.inference_mode()
def restore_face(img: np.ndarray, box, weight: float) -> np.ndarray:
    cx, cy, s = box
    s = int(min(s, H))
    x0 = int(max(0, min(W - s, cx - s / 2)))
    y0 = int(max(0, min(H - s, cy - s / 2)))
    crop = cv2.resize(img[y0:y0 + s, x0:x0 + s], (512, 512), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(crop[..., ::-1].copy()).permute(2, 0, 1)[None].float().div(255).cuda()
    out = MODEL(t)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255
    back = cv2.resize(out.astype(np.float32), (s, s), interpolation=cv2.INTER_CUBIC)
    m = cv2.resize(FACE_MASK, (s, s))[..., None] * weight
    region = img[y0:y0 + s, x0:x0 + s].astype(np.float32)
    img[y0:y0 + s, x0:x0 + s] = np.clip(region * (1 - m) + back * m, 0, 255).astype(np.uint8)
    return img


def handler(job):
    inp = job["input"]
    work = f"/tmp/job-{uuid.uuid4().hex}"
    os.makedirs(work)
    try:
        src, out = f"{work}/in.mp4", f"{work}/out.mp4"
        with open(src, "wb") as f:
            f.write(base64.b64decode(inp["video_base64"]))
        still = cv2.imdecode(np.frombuffer(base64.b64decode(inp["still_base64"]), np.uint8), cv2.IMREAD_COLOR)
        still = cv2.resize(still, (W, H)).astype(np.float32)
        weight = float(min(1.0, max(0.0, inp.get("face_weight", 0.7))))
        crf = str(int(inp.get("crf", 16)))
        fps = probe_fps(src)

        # O ffmpeg já entrega o quadro escalado para 1920x1080 (lanczos, cobrindo e cortando o centro).
        dec = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", src, "-an", "-vf",
             f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=increase,crop={W}:{H}",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE,
        )
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", crf, "-preset", "fast",
             "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE,
        )
        box = None  # (cx, cy, tamanho), suavizado entre quadros
        n = 0
        t = time.time()
        while True:
            buf = dec.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32)
            img = (still * (1 - BG_MASK) + frame * BG_MASK).astype(np.uint8)
            if weight > 0:
                small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (W // 2, H // 2))
                faces = CASCADE.detectMultiScale(small, 1.1, 5, minSize=(75, 75))
                if len(faces):
                    x, y, w, h = [v * 2 for v in max(faces, key=lambda f: f[2] * f[3])]
                    cur = np.array([x + w / 2, y + h / 2 + 0.05 * h, w * 1.8])
                    box = cur if box is None else box * 0.7 + cur * 0.3
                if box is not None:
                    img = restore_face(img, box, weight)
            enc.stdin.write(img.tobytes())
            n += 1
        enc.stdin.close()
        dec.wait()
        if enc.wait() != 0 or n == 0:
            raise RuntimeError(f"ffmpeg falhou ({n} quadros)")
        took = time.time() - t
        print(f"{n} quadros em {took:.1f}s ({n / took:.1f} q/s)", flush=True)

        with open(out, "rb") as f:
            data = f.read()
        return {"video_base64": base64.b64encode(data).decode(), "bytes": len(data), "frames": n,
                "inference_s": took}
    except Exception as e:  # noqa: BLE001 - devolve a mensagem para o app mostrar
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
