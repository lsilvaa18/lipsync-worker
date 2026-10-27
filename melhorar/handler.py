"""
RunPod Serverless handler: melhora um trecho de talking head (InfiniteTalk 832x464 -> 1080p).
  1) amplia o trecho para 1920x1080 (na GPU);
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
import torch.nn.functional as F
from spandrel import ModelLoader

W, H = 1920, 1080
DETECT_EVERY = 3  # a caixa do rosto é suavizada; detectar a cada 3 quadros basta

t0 = time.time()
MODEL = ModelLoader().load_from_file("/models/GFPGANv1.4.pth").cuda().eval()
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print(f"modelo carregado em {time.time() - t0:.1f}s", flush=True)

# Tudo por quadro roda na GPU (a CPU do serverless é fraca): ampliação, fundo, recorte e colagem do rosto.
# Máscara do fundo: faixa central de cima a baixo (cabelo, mãos e cadeira se mexem), borda suave.
_bg = np.zeros((H, W), np.float32)
_bg[:, 440:1500] = 1
BG_MASK = torch.from_numpy(cv2.GaussianBlur(_bg, (0, 0), 35)).cuda()[None, None]

YY, XX = np.mgrid[0:512, 0:512]
FACE_MASK = torch.from_numpy(
    (np.clip(1.2 - np.hypot((XX - 256) / 200, (YY - 270) / 235), 0, 1) ** 0.8).astype(np.float32)
).cuda()[None, None]


def probe(path: str):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", path], capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    num, den = out[2].split("/")
    return int(out[0]), int(out[1]), float(num) / float(den)


def cover_1080(f: torch.Tensor) -> torch.Tensor:
    """Amplia (bicúbico) cobrindo 1920x1080 e corta o centro, como o scale+crop do ffmpeg."""
    h, w = f.shape[-2:]
    k = max(W / w, H / h)
    nw, nh = max(W, round(w * k)), max(H, round(h * k))
    f = F.interpolate(f, size=(nh, nw), mode="bicubic", align_corners=False)
    y, x = (nh - H) // 2, (nw - W) // 2
    return f[..., y:y + H, x:x + W]


@torch.inference_mode()
def restore_face(img: torch.Tensor, box, weight: float) -> torch.Tensor:
    cx, cy, s = box
    s = int(min(s, H))
    x0 = int(max(0, min(W - s, cx - s / 2)))
    y0 = int(max(0, min(H - s, cy - s / 2)))
    region = img[..., y0:y0 + s, x0:x0 + s]
    crop = F.interpolate(region, size=(512, 512), mode="bilinear", antialias=True, align_corners=False)
    out = MODEL(crop.flip(1) / 255).clamp(0, 1).flip(1) * 255  # o modelo quer RGB; o quadro é BGR
    back = F.interpolate(out, size=(s, s), mode="bicubic", align_corners=False)
    m = F.interpolate(FACE_MASK, size=(s, s), mode="bilinear", align_corners=False) * weight
    img[..., y0:y0 + s, x0:x0 + s] = region * (1 - m) + back * m
    return img


@torch.inference_mode()
def handler(job):
    inp = job["input"]
    work = f"/tmp/job-{uuid.uuid4().hex}"
    os.makedirs(work)
    try:
        src, out = f"{work}/in.mp4", f"{work}/out.mp4"
        with open(src, "wb") as f:
            f.write(base64.b64decode(inp["video_base64"]))
        still = cv2.imdecode(np.frombuffer(base64.b64decode(inp["still_base64"]), np.uint8), cv2.IMREAD_COLOR)
        still = torch.from_numpy(cv2.resize(still, (W, H))).cuda().permute(2, 0, 1)[None].float()
        weight = float(min(1.0, max(0.0, inp.get("face_weight", 0.7))))
        crf = str(int(inp.get("crf", 16)))
        w0, h0, fps = probe(src)

        dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                               stdout=subprocess.PIPE)
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", crf, "-preset", "veryfast",
             "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE,
        )
        box = None  # (cx, cy, tamanho), suavizado entre quadros
        n = 0
        t = time.time()
        while True:
            buf = dec.stdout.read(w0 * h0 * 3)
            if len(buf) < w0 * h0 * 3:
                break
            frame = torch.frombuffer(bytearray(buf), dtype=torch.uint8).cuda().view(h0, w0, 3)
            frame = cover_1080(frame.permute(2, 0, 1)[None].float())
            img = still * (1 - BG_MASK) + frame * BG_MASK
            if weight > 0:
                if n % DETECT_EVERY == 0:
                    small = F.interpolate(img, size=(H // 2, W // 2), mode="bilinear", antialias=True, align_corners=False)
                    gray = (small[0, 0] * 0.114 + small[0, 1] * 0.587 + small[0, 2] * 0.299).clamp(0, 255)
                    faces = CASCADE.detectMultiScale(gray.byte().cpu().numpy(), 1.1, 5, minSize=(75, 75))
                    if len(faces):
                        x, y, w, h = [v * 2 for v in max(faces, key=lambda f: f[2] * f[3])]
                        cur = np.array([x + w / 2, y + h / 2 + 0.05 * h, w * 1.8])
                        box = cur if box is None else box * 0.7 + cur * 0.3
                if box is not None:
                    img = restore_face(img, box, weight)
            enc.stdin.write(img.clamp(0, 255).round().byte()[0].permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
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
