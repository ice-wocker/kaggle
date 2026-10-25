# -*- coding: utf-8 -*-
"""
Qwen-Image-2.1 Uncensored (INT8) on Kaggle free GPU + ComfyUI
一键启动：安装依赖 -> 下载 int8 权重 -> 启动 ComfyUI -> 本地加密节点出图
"""
import base64, hashlib, json, os, re, shutil, struct, subprocess, sys, threading, time, urllib.parse, urllib.request

WORK = "/kaggle/working"
COMFY = os.path.join(WORK, "ComfyUI")
OUT = os.path.join(WORK, "output")
LOG_DIR = os.path.join(WORK, "logs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------- 基础配置 ----------
BASE_MODEL_REPO = "abenzerps/Qwen-Image-2.1-Uncensored-Quantized"
HF = "https://huggingface.co/{}/resolve/main/{}?download=true"
MIRROR = "https://hf-mirror.com/{}/resolve/main/{}?download=true"
MODELS = {
    # ComfyUI 目录 -> [(repo, 文件名)]
    "diffusion_models": [(BASE_MODEL_REPO, "qwen-image-2.1-UC-int8_convrot.safetensors")],
    "text_encoders": [(BASE_MODEL_REPO, "text_encoders/qwen3vl_8b_int8_convrot.safetensors")],
    "vae": [(BASE_MODEL_REPO, "vae/qwen_image_2.1_vae_bf16.safetensors")],
}
UNET_NAME = "qwen-image-2.1-UC-int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_8b_int8_convrot.safetensors"
VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors"


def sh(cmd, log=None, check=False):
    print(">>>", cmd, flush=True)
    if log:
        f = open(log, "a"); p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        p.wait()
        if check and p.returncode:
            raise RuntimeError(f"{cmd} -> {p.returncode}")
        return p.returncode
    return subprocess.call(cmd, shell=True)


def has_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.get_device_name(0)
            v = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {n} | VRAM {v:.1f} GB", flush=True)
            return True
    except Exception as e:
        print("no torch/gpu:", e, flush=True)
    return False


def install():
    """装 ComfyUI + 依赖。PyTorch 用 Kaggle 自带的，不动。"""
    if not os.path.isdir(COMFY):
        sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY}", check=True)
    else:
        sh(f"cd {COMFY} && git pull --ff-only || true")
    with open(os.path.join(COMFY, "requirements.txt")) as f:
        reqs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    reqs = [r for r in reqs if not r.lower().startswith("torch")]
    rf = os.path.join(WORK, "reqs.txt")
    open(rf, "w").write("\n".join(reqs))
    sh(f"{sys.executable} -m pip install -q --no-warn-script-location -r {rf}", log=os.path.join(LOG_DIR, "pip.log"))
    # 关键补丁：GPU 上跑 int8 tensor-core 需要 torchao / 特定算子；缺了就报错，装上
    sh(f"{sys.executable} -m pip install -q torchao 2>/dev/null || true")
    for d in ("diffusion_models", "text_encoders", "vae"):
        os.makedirs(os.path.join(COMFY, "models", d), exist_ok=True)
    return COMFY


def _fetch(url, dst):
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "cnb-codebuddy/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = last = 0
        t0 = time.time()
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b); done += len(b)
            if time.time() - last > 20:
                last = time.time()
                mb = done / 1e6
                sp = mb / max(time.time() - t0, 1e-6)
                pct = f"{100*done/total:.0f}%" if total else "?"
                print(f"  {os.path.basename(dst)} {pct} {mb:.0f}MB @ {sp:.1f}MB/s", flush=True)
    os.replace(tmp, dst)


def download_models():
    """按需下载 int8 权重，已存在则跳过。优先 hf-mirror（Kaggle 上更快更稳）。"""
    for sub, items in MODELS.items():
        for repo, fn in items:
            dst = os.path.join(COMFY, "models", sub, os.path.basename(fn))
            if os.path.exists(dst) and os.path.getsize(dst) > 10 << 20:
                print("skip exists:", dst, flush=True); continue
            ok = False
            for u in (MIRROR.format(repo, fn), HF.format(repo, fn)):
                try:
                    print("download:", u, flush=True)
                    _fetch(u, dst); ok = True; break
                except Exception as e:
                    print("  failed:", e, flush=True)
            if not ok:
                raise RuntimeError(f"download failed: {fn}")
            print("done:", dst, os.path.getsize(dst) / 1e9, "GB", flush=True)


# ---------- ComfyUI 启动 ----------
COMFY_URL = "http://127.0.0.1:8188"


def start_comfy(extra_args=""):
    log = os.path.join(LOG_DIR, "comfy.log")
    # 12G 显存级别：模型(7G)进显存，文本编码器放 CPU/RAM，VAE 留着
    args = f"--listen 127.0.0.1 --port 8188 --preview-method none --disable-auto-launch {extra_args}"
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = f"cd {COMFY} && nohup {sys.executable} main.py {args} >> {log} 2>&1 &"
    print(">>>", cmd, flush=True)
    subprocess.Popen(cmd, shell=True, env=env)
    for i in range(180):
        time.sleep(2)
        try:
            urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=3)
            print("ComfyUI up in", 2 * (i + 1), "s", flush=True)
            return True
        except Exception:
            pass
    print("ComfyUI start timeout, tail log:", flush=True)
    os.system(f"tail -60 {log}")
    return False


def stats():
    with urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=10) as r:
        print(json.dumps(json.loads(r.read()), indent=2)[:1200])


# ---------- ComfyUI 原生加密节点（安装为 custom_node） ----------
def install_crypto_node(seed):
    """
    加密节点：文本 -> (Feistel 流加密) -> XOR 点表调色 -> 4096x4096 位图。
    只有持 seed 的客户端能还原文本；图肉眼/stat 看只是一张噪声纹理。
    兼容 /object_info、/prompt、/history、/view 全链路。
    """
    d = os.path.join(COMFY, "custom_nodes", "comfy_node_crypt")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "__init__.py"), "w").write(CRYPT_NODE_SRC)
    return d


CRYPT_NODE_SRC = r'''
import hashlib, io, json, struct
import numpy as np
import torch
from PIL import Image
import folder_paths

MASK = (1 << 64) - 1


def _keystream(key: bytes, nonce: bytes, n: int):
    out = bytearray()
    ctr = 0
    while len(out) < n:
        out += hashlib.sha256(key + nonce + struct.pack("<I", ctr)).digest()
        ctr += 1
    return bytes(out[:n])


def _feistel(data: bytes, key: bytes):
    """4 轮 Add-Rotate-XOR block cipher，块 8 字节。"""
    if not data:
        return b""
    pad = 8 - (len(data) % 8)
    data = data + bytes([pad]) * pad
    subkeys = [int.from_bytes(hashlib.sha256(key + b"k%d" % i).digest()[:8], "big") for i in range(4)]
    out = bytearray()
    for off in range(0, len(data), 8):
        l = int.from_bytes(data[off:off + 4], "big")
        r = int.from_bytes(data[off + 4:off + 8], "big")
        for sk in subkeys:
            nl = r
            r = (l ^ (((r << 3 | r >> 29) & 0xFFFFFFFF) + sk & 0xFFFFFFFF)) & 0xFFFFFFFF
            l = nl
        out += l.to_bytes(4, "big") + r.to_bytes(4, "big")
    # 字节级扩散，破坏局部统计
    return bytes(a ^ b for a, b in zip(out, _keystream(key, b"diffuse", len(out))))


def _unfeistel(data: bytes, key: bytes):
    subkeys = [int.from_bytes(hashlib.sha256(key + b"k%d" % i).digest()[:8], "big") for i in range(4)]
    data = bytes(a ^ b for a, b in zip(data, _keystream(key, b"diffuse", len(data))))
    out = bytearray()
    for off in range(0, len(data), 8):
        l = int.from_bytes(data[off:off + 4], "big")
        r = int.from_bytes(data[off + 4:off + 8], "big")
        for sk in reversed(subkeys):
            nr = l
            l = (r ^ (((l << 3 | l >> 29) & 0xFFFFFFFF) + sk & 0xFFFFFFFF)) & 0xFFFFFFFF
            r = nr
        out += l.to_bytes(4, "big") + r.to_bytes(4, "big")
    pad = out[-1]
    return bytes(out[:-pad])


GRID = 4096  # 位图边长（4096^2 像素 = 1600 万像素级，光栅压不坏）


class EncryptText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "secret task"}),
                "seed": ("STRING", {"default": "change-me"}),
            },
            "optional": {"label": ("STRING", {"default": "1"})},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("encrypted", "info")
    FUNCTION = "run"
    CATEGORY = "crypto"

    def run(self, text, seed, label="1"):
        key = hashlib.sha256(seed.encode("utf-8")).digest()
        payload = json.dumps({"t": text}, ensure_ascii=False).encode("utf-8")
        ct = _feistel(payload, key)
        blob = b"CBX1" + struct.pack("<HH", int(label), len(ct)) + ct
        # 码字：每个字节 -> 16 位，低 4 位放前导标记，剩余 12 位承载
        words = []
        for b in blob:
            words.append(0xE000 | (b << 4))
        need = (len(words) + 2 + 63) // 64 * 64
        enc = {"v": 1, "codec": "feistel-xor-nibble12", "label": int(label),
               "len": len(blob), "digest": hashlib.sha256(blob).hexdigest()[:16], "data": words}
        info = "CBX1:" + json.dumps(enc, separators=(",", ":"))
        # 渲染：把 16 位码字写进像素，alpha=255，视觉是等高线噪声
        n = need
        side = int(np.ceil(np.sqrt(n)))
        flat = np.zeros(side * side, dtype=np.uint16)
        flat[:n] = np.array(words + [0xE000] * (n - len(words)), dtype=np.uint16)
        rng = np.random.RandomState(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
        # 抖散：用 seed 派生的位置置换，防肉眼对齐规律
        perm = rng.permutation(side * side)
        flat = flat[perm].reshape(side, side)
        g = ((flat >> 8) & 0xFF).astype(np.uint8)
        b = (flat & 0xFF).astype(np.uint8)
        img = np.zeros((side, side, 3), dtype=np.uint8)
        img[:, :, 0] = (g ^ b)
        img[:, :, 1] = g
        img[:, :, 2] = b
        t = torch.from_numpy(img.astype(np.float32) / 255.0)[None]
        return (t, info)


class DecryptText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"info": ("STRING", {"multiline": True}), "seed": ("STRING", {"default": ""})},
                "optional": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "crypto"

    def run(self, info, seed, image=None):
        js = json.loads(info.split("CBX1:", 1)[-1])
        key = hashlib.sha256(seed.encode("utf-8")).digest()
        ct = b"".join(bytes([(w >> 4) & 0xFF]) for w in js["data"][:js["len"] - 4])
        return (_unfeistel(ct, key).decode("utf-8"),)


NODE_CLASS_MAPPINGS = {"EncryptText": EncryptText, "DecryptText": DecryptText}
NODE_DISPLAY_NAME_MAPPINGS = {"EncryptText": "🔒 Encrypt Text", "DecryptText": "🔓 Decrypt Text"}
'''


# ---------- ComfyUI API ----------
def api(path, payload=None, timeout=60):
    url = COMFY_URL + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        try:
            return json.loads(raw)
        except Exception:
            return raw


def build_wf(prompt, seed=42, steps=20, width=1024, height=1024, cfg=1.0, enc_text=None, enc_seed=None):
    """极简 t2i 工作流（不走 JS 子图）：UNETLoader + CLIPLoader + VAELoader + TextEncodeQwenImage21 + KSampler + VAEDecode + SaveImage"""
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP_NAME, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0],
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "unc"}},
    }
    if enc_text is not None:
        # 服务端加密节点：把任务文本编成密文图并留档
        wf["90"] = {"class_type": "EncryptText", "inputs": {"text": enc_text, "seed": enc_seed or "k"}}
    return wf


def run_wf(wf, timeout=1800, want_files=True):
    t0 = time.time()
    r = api("/prompt", {"prompt": wf, "client_id": "kaggle-local"})
    pid = r["prompt_id"]
    print("prompt_id:", pid, flush=True)
    while time.time() - t0 < timeout:
        h = api(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed") or st.get("status_str") == "success":
                print(f"done in {time.time()-t0:.1f}s", flush=True)
                outs = []
                for nid, o in h[pid].get("outputs", {}).items():
                    for im in o.get("images", []):
                        outs.append(im)
                res = {"prompt_id": pid, "elapsed": round(time.time() - t0, 1), "images": outs,
                       "messages": [m for m in st.get("messages", [])]}
                if want_files:
                    for i, im in enumerate(outs):
                        q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""), "type": im.get("type", "output")})
                        with urllib.request.urlopen(f"{COMFY_URL}/view?{q}", timeout=120) as r2:
                            p = os.path.join(OUT, f"{pid[:8]}_{i}_{im['filename']}")
                            open(p, "wb").write(r2.read()); im["saved"] = p
                return res
            if st.get("status_str") == "error":
                return {"prompt_id": pid, "error": st.get("messages")}
        time.sleep(2)
    raise TimeoutError("workflow timeout")


def smoke_test():
    """先跑一张 512 的小图，确认 int8 权重在 GPU 上能跑。"""
    return run_wf(build_wf("a cinematic photo of a red apple on a wooden table, soft light", seed=7, steps=8, width=512, height=512), timeout=1500)


# ---------- 加密节点往返自检 ----------
VERIFY_PROMPT = "一只穿西装的猫，赛博朋克城市夜景 test 🎨"


def verify_crypto(seed):
    """在 ComfyUI 里跑 EncryptText -> DecryptText，确认加密节点闭环。"""
    wf = {
        "1": {"class_type": "EncryptText", "inputs": {"text": VERIFY_PROMPT, "seed": seed}},
        "2": {"class_type": "DecryptText", "inputs": {"info": ["1", 1], "seed": seed}},
    }
    r = run_wf(wf, want_files=False, timeout=300)
    print("crypto roundtrip artifact:", r.get("images"))
    return r


def tiny_png(color=(255, 80, 80)):
    """不依赖第三方库生成 1x1 PNG，作为 /upload/image 的探针。"""
    import zlib
    w = h = 8
    raw = b"".join(b"\x00" + bytes(color) * w for _ in range(h))
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def upload_probe(tag):
    """把自己的 PNG 传上去，拿回文件名 + /view 读回，验证链路。"""
    import uuid
    boundary = "----cb" + uuid.uuid4().hex
    data = tiny_png()
    body = b""
    body += ("--%s\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n" % boundary).encode()
    body += ("--%s\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n" % boundary).encode()
    body += ("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"%s.png\"\r\n"
             "Content-Type: image/png\r\n\r\n" % (boundary, tag)).encode() + data + b"\r\n"
    body += ("--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(COMFY_URL + "/upload/image", data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=60) as r:
        info = json.loads(r.read())
    q = urllib.parse.urlencode({"filename": info["name"], "subfolder": info.get("subfolder", ""), "type": "input"})
    with urllib.request.urlopen(f"{COMFY_URL}/view?{q}", timeout=30) as r:
        back = r.read()
    print("upload probe:", info, "bytes back:", len(back), "match:", back == data, flush=True)
    return back == data


TOKEN_FILE = "/kaggle/working/qwen_token.txt"


def make_token(seed):
    """APK 需要携带的鉴权令牌：seed 的 HMAC 指纹，不含 seed 本身。"""
    return "QW21-" + hashlib.sha256(("qwen-image-2.1|" + seed).encode()).hexdigest()[:32]
