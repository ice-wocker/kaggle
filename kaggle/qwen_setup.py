# -*- coding: utf-8 -*-
"""
Qwen-Image-2.1 Uncensored (INT8) on Kaggle free GPU + ComfyUI
一键启动：安装依赖 -> 下载 int8 权重 -> 启动 ComfyUI -> 本地加密节点出图
"""
import base64, hashlib, json, math, os, re, shutil, struct, subprocess, sys, threading, time
import urllib.error, urllib.parse, urllib.request

import numpy as np

# /kaggle/working 只有 ~20G，放不下 ComfyUI + 16G 权重；
# 大件放 /tmp（Kaggle 上空间更宽），working 只留结果和日志。
WORK = "/kaggle/working"
TMP = "/tmp/qwen21"
COMFY = os.path.join(TMP, "ComfyUI")
OUT = os.path.join(WORK, "output")
LOG_DIR = os.path.join(TMP, "logs")
for d in (TMP, OUT, LOG_DIR):
    os.makedirs(d, exist_ok=True)

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
    """跑命令；log 指定时把 stdout+stderr 一并落盘，失败抛出日志尾部。"""
    print(">>>", cmd, flush=True)
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a") as f:
            f.write(f"\n===== {time.strftime('%H:%M:%S')} {cmd}\n")
            f.flush()
            rc = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        if check and rc:
            raise RuntimeError(f"{cmd} -> rc={rc}\n" + tail_log(log))
        return rc
    rc = subprocess.call(cmd, shell=True)
    if check and rc:
        raise RuntimeError(f"{cmd} -> rc={rc}")
    return rc


def mark(msg):
    """把进度写进 output，失败后也能从产物里看出卡在哪。"""
    try:
        with open(os.path.join(OUT, "progress.log"), "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass
    print("[progress]", msg, flush=True)


def tail_log(path, n=60):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 20000))
            return f.read().decode("utf-8", "replace")[-4000:]
    except Exception:
        return ""


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
    """装 ComfyUI + 依赖。PyTorch 用 Kaggle 自带的，不重装。"""
    mark("install: 开始")
    if not os.path.isdir(COMFY):
        sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY}",
           log=os.path.join(LOG_DIR, "setup.log"), check=True)
    mark("install: clone 完成")

    with open(os.path.join(COMFY, "requirements.txt")) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    # 只跳过镜像自带的 torch / torchvision（重装会拉 CUDA 轮子覆盖环境）。
    # torchsde、torchvision 之外的小包照装 —— 少装一个就起不来。
    def _preinstalled(line):
        name = re.split(r"[=<>\[!~ ]", line.lower(), 1)[0]
        return name in ("torch", "torchvision")
    reqs = [l for l in lines if not _preinstalled(l)]
    rf = os.path.join(WORK, "reqs.txt")
    open(rf, "w").write("\n".join(reqs) + "\n")
    print(f"deps: {len(reqs)} 条（跳过 {len(lines) - len(reqs)} 条 torch*）", flush=True)

    mark(f"install: 开始装 {len(reqs)} 条依赖")
    pip_log = os.path.join(LOG_DIR, "pip.log")
    rc = sh(f"{sys.executable} -m pip install --no-warn-script-location -r {rf}", log=pip_log)
    if rc:
        # 逐条重试，让单条坏依赖不拖死整批
        print("批量安装失败，逐条重试", flush=True)
        bad = []
        for r in reqs:
            if sh(f"{sys.executable} -m pip install -q --no-warn-script-location '{r}'", log=pip_log):
                bad.append(r)
        if bad:
            print("装不上的依赖（继续跑，运行时再看）:", bad, flush=True)

    # 启动前的依赖自检：缺模块提前补，别等 ComfyUI 跑到一半才炸
    need = ["torchsde", "einops", "transformers", "safetensors", "aiohttp",
            "yaml", "PIL", "scipy", "tqdm", "psutil", "alembic", "sqlalchemy", "filelock"]
    import importlib
    missing = []
    for m in need:
        try:
            importlib.import_module(m)
        except Exception:
            missing.append(m)
    if missing:
        mark(f"install: 补装缺失模块 {missing}")
        pkg = {"yaml": "pyyaml", "PIL": "pillow", "sqlalchemy": "sqlalchemy"}
        sh(f"{sys.executable} -m pip install -q --no-warn-script-location "
           + " ".join(pkg.get(m, m) for m in missing), log=os.path.join(LOG_DIR, "pip.log"))

    for d in ("diffusion_models", "text_encoders", "vae"):
        os.makedirs(os.path.join(COMFY, "models", d), exist_ok=True)
    mark("install: 完成")
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
    # 节点文件 = 模板 + 模块级算法，保证两处永远同源
    node_src = CRYPT_NODE_SRC_TEMPLATE.replace("@@ALGO@@", ALGO_SRC)
    open(os.path.join(d, "__init__.py"), "w").write(node_src)
    mark(f"crypto-node: 已写入 {len(node_src)} 字节")
    return d


# ================== 密文图算法（模块级，节点与自检共用） ==================
MIN_SIDE = 32  # 密文图最小边长；码字多时自动长大，保持接近 100% 的填充率
FILL = 0xE000  # 填充码字：高 4 位标记位，不会与真实数据混淆


def _keystream(key, nonce, n):
    out = bytearray()
    ctr = 0
    while len(out) < n:
        out += hashlib.sha256(key + nonce + struct.pack("<I", ctr)).digest()
        ctr += 1
    return bytes(out[:n])


def _feistel(data, key):
    """4 轮 ARX（加-转-XOR）块密码，8 字节分组，PKCS#7 填充。"""
    pad = 8 - (len(data) % 8)
    data = data + bytes([pad]) * pad
    subkeys = [int.from_bytes(hashlib.sha256(key + b"k%d" % i).digest()[:4], "big") for i in range(4)]
    out = bytearray()
    for off in range(0, len(data), 8):
        l = int.from_bytes(data[off:off + 4], "big")
        r = int.from_bytes(data[off + 4:off + 8], "big")
        for sk in subkeys:
            nl = r
            r = (l ^ (((r << 3 | r >> 29) & 0xFFFFFFFF) + sk & 0xFFFFFFFF)) & 0xFFFFFFFF
            l = nl
        out += l.to_bytes(4, "big") + r.to_bytes(4, "big")
    return bytes(a ^ b for a, b in zip(out, _keystream(key, b"diffuse", len(out))))


def _unfeistel(data, key):
    subkeys = [int.from_bytes(hashlib.sha256(key + b"k%d" % i).digest()[:4], "big") for i in range(4)]
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
    return bytes(out[:-out[-1]])



def _side_for(count):
    """
    码字数 -> 画布边长。边长取 2 的幂（从 MIN_SIDE 起翻倍）：
    这样 side*side 也是 2 的幂，位反转才能构成严格双射，铺满且不重位。
    """
    side = MIN_SIDE
    while side * side < count:
        side *= 2
    return side


def _slot(i, side):
    """
    码字序号 -> 像素线性下标。对 side*side 个格子做位反转（Van der Corput）：
    任意前缀都铺得最均匀。side 是 2 的幂 => bit_length-1 位刚好覆盖全空间。
    """
    bits = (side * side).bit_length() - 1
    x = i % (side * side)
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def render(words):
    """码字列表 -> side x side x 3 uint8 图。三通道互异，肉眼是彩色噪声。"""
    side = _side_for(len(words))
    n = side * side
    flat = np.full(n, FILL, dtype=np.uint16)
    for i, w in enumerate(words):
        if i < n:
            flat[_slot(i, side)] = w
    g = ((flat >> 8) & 0xFF).astype(np.uint8).reshape(side, side)
    b = (flat & 0xFF).astype(np.uint8).reshape(side, side)
    img = np.empty((side, side, 3), dtype=np.uint8)
    img[:, :, 0] = g ^ b
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def unrender(img, count):
    """图 -> 前 count 个码字。客户端 Crypto.java 用同一公式。"""
    side = img.shape[0]
    a = img.reshape(-1, 3).astype(np.uint16)
    flat = (a[:, 1] << 8) | a[:, 2]
    return [int(flat[_slot(i, side)]) for i in range(count)]


def image_to_tensor(img):
    return torch.from_numpy(img.astype(np.float32) / 255.0)[None]


def encode_words(seed, text):
    """明文 -> 码字数组，与 EncryptText 节点同构（给自检用）。"""
    key = hashlib.sha256(seed.encode("utf-8")).digest()
    ct = _feistel(text.encode("utf-8"), key)
    blob = b"\xCB\x21\x01\x00" + len(ct).to_bytes(2, "big") + ct
    return [FILL | (b << 4) for b in blob]


def decode_words(words, seed):
    """码字数组 -> 明文（给自检用）。"""
    blob = bytes((w >> 4) & 0xFF for w in words)
    if blob[:2] != b"\xCB\x21":
        raise ValueError("密文头不对")
    n = int.from_bytes(blob[4:6], "big")
    key = hashlib.sha256(seed.encode("utf-8")).digest()
    return _unfeistel(blob[6:6 + n], key).decode("utf-8")
# ========================================================================

# 节点文件里要用的算法源码：显式写一份，避免 inspect 在 notebook 里取不到源码
ALGO_SRC = '''
import math

FILL = {fill}
MIN_SIDE = {min_side}


def _keystream(key, nonce, n):
    out = bytearray()
    ctr = 0
    while len(out) < n:
        out += hashlib.sha256(key + nonce + struct.pack("<I", ctr)).digest()
        ctr += 1
    return bytes(out[:n])


def _feistel(data, key):
    """4 轮 ARX（加-转-XOR）块密码，8 字节分组，PKCS#7 填充。"""
    pad = 8 - (len(data) % 8)
    data = data + bytes([pad]) * pad
    subkeys = [int.from_bytes(hashlib.sha256(key + b"k%d" % i).digest()[:4], "big") for i in range(4)]
    out = bytearray()
    for off in range(0, len(data), 8):
        l = int.from_bytes(data[off:off + 4], "big")
        r = int.from_bytes(data[off + 4:off + 8], "big")
        for sk in subkeys:
            nl = r
            r = (l ^ (((r << 3 | r >> 29) & 0xFFFFFFFF) + sk & 0xFFFFFFFF)) & 0xFFFFFFFF
            l = nl
        out += l.to_bytes(4, "big") + r.to_bytes(4, "big")
    return bytes(a ^ b for a, b in zip(out, _keystream(key, b"diffuse", len(out))))


def _unfeistel(data, key):
    subkeys = [int.from_bytes(hashlib.sha256(key + b"k%d" % i).digest()[:4], "big") for i in range(4)]
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
    return bytes(out[:-out[-1]])


def _side_for(count):
    side = MIN_SIDE
    while side * side < count:
        side *= 2
    return side


def _slot(i, side):
    bits = (side * side).bit_length() - 1
    x = i % (side * side)
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def render(words):
    side = _side_for(len(words))
    n = side * side
    flat = np.full(n, FILL, dtype=np.uint16)
    for i, w in enumerate(words):
        if i < n:
            flat[_slot(i, side)] = w
    g = ((flat >> 8) & 0xFF).astype(np.uint8).reshape(side, side)
    b = (flat & 0xFF).astype(np.uint8).reshape(side, side)
    img = np.empty((side, side, 3), dtype=np.uint8)
    img[:, :, 0] = g ^ b
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def unrender(img, count):
    side = img.shape[0]
    a = img.reshape(-1, 3).astype(np.uint16)
    flat = (a[:, 1] << 8) | a[:, 2]
    return [int(flat[_slot(i, side)]) for i in range(count)]


def image_to_tensor(img):
    return torch.from_numpy(img.astype(np.float32) / 255.0)[None]
'''.format(fill=hex(FILL), min_side=MIN_SIDE)


CRYPT_NODE_SRC_TEMPLATE = r'''
import hashlib, json, struct
import numpy as np
import torch


# ===== 共享算法（与模块级 ALGO 同源，构建时注入）=====
@@ALGO@@


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
        ct = _feistel(text.encode("utf-8"), key)
        # 头：CB 21 01 00 | len(2, big-endian) | 密文。与 APK 的 Crypto.encodeTask 一致
        blob = b"\xCB\x21\x01\x00" + len(ct).to_bytes(2, "big") + ct
        # 每个字节 -> 16 位码字：高 4 位留标记，低 12 位承载（8 位数据 + 4 位冗余）
        words = [FILL | (b << 4) for b in blob]
        enc = {"v": 3, "codec": "feistel-arx-bitrev", "label": int(label),
               "side": _side_for(len(words)),
               "len": len(words), "digest": hashlib.sha256(blob).hexdigest()[:16]}
        info = "CBX1:" + json.dumps(enc, separators=(",", ":"))
        return (image_to_tensor(render(words)), info)


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
        if image is not None:
            # 从像素还原码字：图是唯一载体
            arr = (image[0].cpu().numpy() * 255.0).round().astype(np.uint8)
            words = unrender(arr, js["len"])
        else:
            raise ValueError("需要密文图才能解密")
        ct = bytes((w >> 4) & 0xFF for w in words[6:js["len"]])
        return (_unfeistel(ct, key).decode("utf-8"),)


NODE_CLASS_MAPPINGS = {"EncryptText": EncryptText, "DecryptText": DecryptText}
NODE_DISPLAY_NAME_MAPPINGS = {"EncryptText": "🔒 Encrypt Text", "DecryptText": "🔓 Decrypt Text"}'''

CRYPT_NODE_SRC = CRYPT_NODE_SRC_TEMPLATE  # 安装时用 ALGO 填充，见 install_crypto_node



# ---------- ComfyUI API ----------
def api(path, payload=None, timeout=60):
    """调 ComfyUI HTTP API；4xx/5xx 时把响应体一起抛出来，否则排错全靠猜。"""
    url = COMFY_URL + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"{path} -> HTTP {e.code}: {body[:3000]}") from None
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
        # 加密节点挂进主图：EncryptText 的密文图一起存盘留档，
        # 这样它属于 SaveImage 的上游，不会被判成孤立节点
        wf["90"] = {"class_type": "EncryptText", "inputs": {"text": enc_text, "seed": enc_seed or "k"}}
        wf["91"] = {"class_type": "SaveImage", "inputs": {"images": ["90", 0], "filename_prefix": "cipher"}}
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
    """
    跑 EncryptText -> DecryptText，并把解密结果和密文图都落到 output。
    ComfyUI 要求工作流至少有一个输出节点，所以挂 SaveImage 收加密图。
    """
    wf = {
        "1": {"class_type": "EncryptText", "inputs": {"text": VERIFY_PROMPT, "seed": seed}},
        "2": {"class_type": "DecryptText", "inputs": {"info": ["1", 1], "seed": seed}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "cipher"}},
    }
    r = run_wf(wf, timeout=300)
    print("crypto roundtrip artifact:", json.dumps(r.get("images"), ensure_ascii=False), flush=True)
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
