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
# Kaggle 保存版本时 output 有 ~20GB 上限，超了整次保存直接失败、日志也就没了。
# 权重、ComfyUI、日志全在 /tmp（本地盘，不参与保存）；working 里只留几 KB 的进度文件。
# 代价：会话重启后要重新下 17GB（hf-mirror 约 2 分钟），换来自续跑能可靠落地。
CACHE = os.environ.get("QWEN21_CACHE") or WORK
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
# 当前会话公布的隧道地址、出口 IP、GPU 名：常驻心跳要重播它们，所以放在全局
SERVE_URL = [""]
SERVE_IP = [""]
GPU_NAME_GLOBAL = ["?"]

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


def shout(msg):
    """只打印、不上报的进度行。

    日志是端上唯一的眼睛：它只认 QWEN21:// 里的 stage。以前 _fetch 每 20 秒
    才打一行下载进度，git clone / aiohttp 往磁盘写的那段则是彻底的哑巴，
    手机上看到的就是「等待云端日志」一秒一秒往上加、直到 1200 秒超时。
    这里让云端在任何长耗时动作期间都别超过一两分钟不吭声。
    """
    print(f"[progress] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def mark(msg):
    """进度落盘，失败后也能从产物里看出卡在哪。"""
    try:
        with open(os.path.join(WORK, "qwen_progress.log"), "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass
    publish("stage", msg)
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
    # Kaggle 镜像自带 torch/numpy/transformers/aiohttp/pillow/safetensors/einops，
    # 所以不做 venv、不碰 torch：自定义节点只 import numpy/torch/PIL，用系统 Python 就够。
    # 重装 torch 会拉 CUDA 轮子覆盖镜像里的匹配版本，是上次最容易炸的一环。
    shout("clone ComfyUI")
    sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY}",
       log=os.path.join(LOG_DIR, "setup.log"), check=False)
    if not os.path.isdir(os.path.join(COMFY, "comfy")):
        raise RuntimeError("ComfyUI 拉取失败\n" + tail_log(os.path.join(LOG_DIR, "setup.log")))
    mark("install: clone 完成")

    with open(os.path.join(COMFY, "requirements.txt")) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # 只跳过 torch（镜像自带 cu12x 轮子，重装会覆盖成别的 CUDA 版本）。
    # 别的一律照装 —— 包括 comfy-aimdo / comfy-kitchen 这种带 C 扩展的小包：
    # main.py 第 79 行 import comfy_aimdo.control 是无保护的硬 import，
    # 少一个就整进程起不来，症状是「起 ComfyUI 超时 + ModuleNotFoundError」。
    def _is_torch(line):
        name = re.split(r"[=<>\[!~ ]", line.lower(), 1)[0]
        return name in ("torch", "torchvision")

    reqs = [l for l in lines if not _is_torch(l)]
    open(os.path.join(WORK, "reqs.txt"), "w").write("\n".join(reqs) + "\n")
    print(f"deps: {len(reqs)} 条（跳过 {len(lines) - len(reqs)} 条 torch*）", flush=True)

    # 以前这里只查一张手写的 13 项清单，清单漏了谁谁就永远装不上：
    # 镜像里没 comfy_aimdo，清单里也没它，于是每次都在 start-comfy 炸。
    # 现在以 ComfyUI 自己的 requirements.txt 为准，逐个 import 探活。
    import importlib.util
    # 发行包名和导入名不一致的，查这张表；表里没有的按「包名里的 - 换成 _」猜，
    # 猜错只会多装一次，不会漏装。
    NAMES = {"pyyaml": "yaml", "pillow": "PIL", "pydantic-settings": "pydantic_settings",
             "opencv-python": "cv2", "comfyui-frontend-package": "comfyui_frontend_package",
             "comfyui-workflow-templates": "comfyui_workflow_templates",
             "comfyui-embedded-docs": "comfyui_embedded_docs", "blake3": "blake3",
             "simpleeval": "simpleeval", "av": "av", "comfy-angle": "comfy_angle"}

    def _probe(line):
        """返回 (导入名, 是否已装)。"""
        raw = re.split(r"[=<>\[!~ ;]", line, 1)[0].strip().lower()
        mod = NAMES.get(raw) or raw.replace("-", "_")
        try:
            return mod, importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError, AttributeError):
            return mod, False

    missing = []
    for line in reqs:
        mod, ok = _probe(line)
        if not ok:
            missing.append(line)

    # 包名里有 >= 的照原样喂给 pip，但别丢进日志行——那会把 markdown 弄乱。
    names = [m.split("=")[0].split(">")[0].split("<")[0].strip() for m in missing]
    pip_log = os.path.join(LOG_DIR, "pip.log")
    if missing:
        mark(f"install: 补装 ComfyUI 要求里缺的 {len(missing)} 项，含 comfy-aimdo/comfy-kitchen 这类硬依赖")
        print("缺:", names, flush=True)
        shout(f"pip 装 {len(missing)} 个包：" + " ".join(names[:8]))
        sh(f"{sys.executable} -m pip install -q --no-warn-script-location " + " ".join(missing),
           log=pip_log)
        still = [line for line in missing if not _probe(line)[1]]
        if still:
            # 静默失败最贵：什么都不说，然后 6 分钟后报「启动超时」。
            raise RuntimeError("依赖装完仍导不进: %s\n" % [s.split("=")[0] for s in still]
                               + tail_log(pip_log))
    else:
        mark("install: 依赖已齐，跳过 pip")

    for d in ("diffusion_models", "text_encoders", "vae"):
        os.makedirs(os.path.join(COMFY, "models", d), exist_ok=True)
    mark("install: 完成")
    return COMFY


def _fetch(url, dst, chunk=8 << 20):
    """单文件下载，断点续传：中途断了从 .part 续，不从头再来。"""
    tmp = dst + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"User-Agent": "cnb-codebuddy/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        if r.status == 206:
            total += have
        elif have:
            have = 0  # 服务端不认 Range，重头来
        with open(tmp, "ab" if have else "wb") as f:
            done, last, t0 = have, 0.0, time.time()
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b); done += len(b)
                if time.time() - last > 10:
                    last = time.time()
                    mb = done / 1e6
                    sp = (done - have) / 1e6 / max(time.time() - t0, 1e-6)
                    pct = f"{100*done/total:.0f}%" if total else "?"
                    shout(f"下载 {os.path.basename(dst)} {pct} {mb:.0f}MB @ {sp:.1f}MB/s")
    size = os.path.getsize(tmp)
    if total and size < total:
        raise IOError(f"不完整 {size}/{total}")
    os.replace(tmp, dst)


def _one(sub, repo, fn):
    dst = os.path.join(COMFY, "models", sub, os.path.basename(fn))
    if os.path.exists(dst) and os.path.getsize(dst) > 10 << 20:
        print("skip exists:", dst, flush=True)
        return dst
    for _ in range(3):
        for u in (MIRROR.format(repo, fn), HF.format(repo, fn)):
            try:
                print("download:", u, flush=True)
                _fetch(u, dst)
                print("done:", dst, os.path.getsize(dst) / 1e9, "GB", flush=True)
                return dst
            except Exception as e:
                print("  retry after error:", e, flush=True)
        time.sleep(2)
    raise RuntimeError(f"download failed: {fn}")


def download_models():
    """三个权重并行拉：16G 串行要十几分钟，并行能把这段压掉一大半。"""
    jobs = [(sub, repo, fn) for sub, items in MODELS.items() for repo, fn in items]
    shout(f"开始拉 {len(jobs)} 件权重（合计约 17GB，走 hf-mirror）")
    errs = []

    def run(job):
        try:
            dst = _one(*job)
            shout(f"权重就绪 {os.path.basename(dst)}")
        except Exception as e:  # noqa: BLE001 - 汇总后统一抛
            errs.append(e)

    ts = [threading.Thread(target=run, args=(j,), daemon=True) for j in jobs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0]
    mark("download: 三件权重就绪")


# ---------- ComfyUI 启动 ----------
COMFY_URL = "http://127.0.0.1:8188"


def start_comfy(extra_args=""):
    log = os.path.join(LOG_DIR, "comfy.log")
    # 12G 显存级别：模型(7G)进显存，文本编码器放 CPU/RAM，VAE 留着
    args = f"--listen 0.0.0.0 --port 8188 --preview-method none --disable-auto-launch {extra_args}"
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = f"cd {COMFY} && nohup {sys.executable} main.py {args} >> {log} 2>&1 &"
    print(">>>", cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)
    write_token_file()
    for i in range(180):
        time.sleep(2)
        try:
            urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=3)
            print("ComfyUI up in", 2 * (i + 1), "s", flush=True)
            return True
        except Exception:
            pass
        # 每 30 秒说一句：这段时间最长 6 分钟没日志，端上会以为卡死
        if i and i % 15 == 0:
            shout(f"等 ComfyUI 起来 · {2 * i}s")
        # 进程已经死了就别再等那 6 分钟：直接把真错误报出来。
        # 上一版没这一条，ModuleNotFoundError 被埋成「启动超时」，日志里全是 alembic 噪音。
        if proc.poll() is not None:
            break
    print("ComfyUI start failed/timeout, tail log:", flush=True)
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
    # 认的令牌有两个：REV 里写死的稳定值（APK 内置同一份），和 seed 派生值。
    # 端上手上是哪个都能进——鉴权只为挡住乱敲的人，不为难自己。
    node_src = (CRYPT_NODE_SRC_TEMPLATE.replace("@@ALGO@@", ALGO_SRC)
                .replace("@@TOKENS@@", repr(sorted(CLOUD_TOKENS)))
                .replace("@@SEED@@", seed)
                .replace("@@UNET@@", UNET_NAME)
                .replace("@@CLIP@@", CLIP_NAME)
                .replace("@@VAE@@", VAE_NAME))
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


# ================== 任务信封：把出图参数一起塞进密文 ==================
# 上一版只加密了提示词，步数/尺寸/种子/负面词还是明文 JSON 过网。
# 现在整条指令走同一个信封，网络上只有密码学噪声。
# 布局：CB 22 01 00 | len(2) | ct，ct = Feistel(inner)
# inner：b"Q2" | n(1) x [keylen(1) key value(2) typ(1) payload]
def pack_task(fields, seed):
    """fields: [(k, v)]，v 为 str/int/float/bool。"""
    key = hashlib.sha256(seed.encode("utf-8")).digest()
    inner = bytearray(b"Q2")
    for k, v in fields:
        kb = k.encode("utf-8")
        if isinstance(v, bool):
            typ, raw = 0, b"\x01" if v else b"\x00"
        elif isinstance(v, int):
            typ = 1 if v >= 0 else 2
            raw = (v if v >= 0 else -v).to_bytes(8, "big")
        elif isinstance(v, float):
            typ, raw = 3, struct.pack(">d", v)
        else:
            typ, raw = 4, str(v).encode("utf-8")
        inner += bytes([len(kb)]) + kb + struct.pack(">H", len(raw)) + bytes([typ]) + raw
    ct = _feistel(bytes(inner), key)
    return b"\xCB\x22\x01\x00" + len(ct).to_bytes(2, "big") + ct


def unpack_task(blob, seed):
    key = hashlib.sha256(seed.encode("utf-8")).digest()
    if blob[:2] != b"\xCB\x22":
        raise ValueError("信封头不对")
    n = int.from_bytes(blob[4:6], "big")
    inner = _unfeistel(blob[6:6 + n], key)
    if inner[:2] != b"Q2":
        raise ValueError("信封体不对")
    out, i = {}, 2
    while i < len(inner):
        kl = inner[i]; i += 1
        k = inner[i:i + kl].decode("utf-8"); i += kl
        vl = int.from_bytes(inner[i:i + 2], "big"); i += 2
        typ = inner[i]; i += 1
        raw = inner[i:i + vl]; i += vl
        if typ == 0:
            out[k] = raw == b"\x01"
        elif typ == 1:
            out[k] = int.from_bytes(raw, "big")
        elif typ == 2:
            out[k] = -int.from_bytes(raw, "big")
        elif typ == 3:
            out[k] = struct.unpack(">d", raw)[0]
        else:
            out[k] = raw.decode("utf-8")
    return out
# ========================================================================

# 节点文件里要用的算法源码：显式写一份，避免 inspect 在 notebook 里取不到源码
ALGO_SRC = '''
# 节点文件只 import 了 numpy/torch/json，哈希与打包要在这里自带，
# 否则节点加载时会在第一行解密的 _keystream 里炸掉。
import hashlib, struct

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
import base64, hashlib, json, struct, uuid
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
NODE_DISPLAY_NAME_MAPPINGS = {"EncryptText": "🔒 Encrypt Text", "DecryptText": "🔓 Decrypt Text"}


# ===== 手机端专用路由：探活 + 一次性提交加密任务 =====
# 这层只是「解信封 + 拼工作流」，解密后的提示词从不离开进程内存，
# 也不打印、不落盘；网络上永远是 CB 22 信封。
# 注意：except 是裸的 —— 路由装不上也要把加密节点留住，
# 节点本身才是这个仓库的主角。
try:
    from server import PromptServer
    from aiohttp import web
    import server as _srv
    import execution as _exe
    import time as _time
    import traceback as _tb

    _JOBS = {}
    _TASKS = {}

    def _unpack(blob):
        key = hashlib.sha256(b"@@SEED@@").digest()
        if blob[:2] != b"\xCB\x22":
            raise ValueError("bad envelope")
        n = int.from_bytes(blob[4:6], "big")
        inner = _unfeistel(blob[6:6 + n], key)
        if inner[:2] != b"Q2":
            raise ValueError("bad body")
        out, i = {}, 2
        while i < len(inner):
            kl = inner[i]; i += 1
            k = inner[i:i + kl].decode("utf-8"); i += kl
            vl = int.from_bytes(inner[i:i + 2], "big"); i += 2
            typ = inner[i]; i += 1
            raw = inner[i:i + vl]; i += vl
            if typ == 0:
                out[k] = raw == b"\x01"
            elif typ == 1:
                out[k] = int.from_bytes(raw, "big")
            elif typ == 2:
                out[k] = -int.from_bytes(raw, "big")
            elif typ == 3:
                out[k] = struct.unpack(">d", raw)[0]
            else:
                out[k] = raw.decode("utf-8")
        return out

    def _build(f):
        w = int(f.get("width", 1024)) // 64 * 64
        h = int(f.get("height", 1024)) // 64 * 64
        g = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "@@UNET@@", "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "@@CLIP@@", "type": "qwen_image", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "@@VAE@@"}},
            "4": {"class_type": "TextEncodeQwenImage21",
                  "inputs": {"clip": ["2", 0], "prompt": f.get("prompt", ""),
                             "negative_prompt": f.get("negative", ""), "resolution": 1024}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": w, "height": h, "batch_size": int(f.get("batch", 1))}},
            "6": {"class_type": "KSampler", "inputs": {
                "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0],
                "seed": int(f.get("seed", 0)), "steps": int(f.get("steps", 20)),
                "cfg": float(f.get("cfg", 1.0)), "sampler_name": "euler", "scheduler": "simple",
                "denoise": 1.0}},
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "q21"}},
        }
        return g

    TOKENS = set(@@TOKENS@@)

    def _given(request, body):
        # body["t"] 是信封本体，不是凭据：令牌只认请求头、查询串或 body["k"]
        h = request.headers
        return (h.get("X-QWEN-TOKEN")
                or (h.get("Authorization") or "").replace("Bearer ", "").replace("bearer ", "").strip()
                or request.query.get("token")
                or (body or {}).get("k") or "")

    def _auth(request, body):
        return _given(request, body) in TOKENS

    @PromptServer.instance.routes.get("/qwen/health")
    async def _qwen_health(request):
        # ok 是"服务活着"（不带令牌也看得出来），auth 才是"令牌对上没有"。
        # 以前只回一个 ok，端上探活失败分不清是服务没醒还是令牌过期。
        ok = _auth(request, None)
        return web.json_response({"ok": True, "node": "comfy_node_crypt", "auth": ok,
                                  "queue": len(_srv.PromptServer.instance.prompt_queue.get_current_queue()[0])})

    @PromptServer.instance.routes.post("/qwen/generate")
    async def _qwen_generate(request):
        body = await request.json()
        if not _auth(request, body):
            return web.json_response({"error": "unauthorized", "got": len(_given(request, body))}, status=401)
        raw = base64.b64decode(body.get("t") or b"")
        try:
            f = _unpack(raw)
        except Exception as e:
            return web.json_response({"error": "bad envelope: %s" % e}, status=400)
        wf = _build(f)
        pid = str(uuid.uuid4())
        _TASKS[pid] = {"t0": _time.time(),
                       "shape": "%sx%s/%s步/%s张" % (f.get("width"), f.get("height"),
                                                     f.get("steps"), f.get("batch", 1)),
                       "tip": ""}
        _srv.PromptServer.instance.prompt_queue.put((0, pid, wf))
        return web.json_response({"id": pid})

    @PromptServer.instance.routes.get("/qwen/task/{tid}")
    async def _qwen_task(request):
        tid = request.match_info["tid"]
        info = _TASKS.get(tid) or {}
        q = _srv.PromptServer.instance.prompt_queue
        try:
            cur = q.get_current_queue()
            running = any(tid == i[1] for i in cur[0])
            pending = any(tid == i[1] for i in cur[1])
        except Exception:
            running = pending = False
        try:
            hist = _srv.PromptServer.instance.prompt_queue.get_history(tid)
        except Exception:
            hist = {}
        if not hist:
            waited = _time.time() - info.get("t0", _time.time())
            # 在跑却三分钟没产出，最典型的原因是 T4 显存不够、模型退回 CPU 硬算：
            # 手机端看到的只是"转圈"，说实话比干等有用。
            stuck = running and waited > 180 and not info.get("tip")
            if stuck:
                info["tip"] = "GPU 长时间无输出：16G 显存可能不够 1024x1024，试 768x768 或把张数改成 1"
            return web.json_response({"state": "running" if running else "queued",
                                      "queue": len(cur[0]) if running else 0,
                                      "elapsed": round(waited, 1),
                                      "shape": info.get("shape", ""),
                                      "tip": info.get("tip", "")})
        e = hist[tid]
        st = e.get("status") or {}
        if st.get("status_str") == "error":
            return web.json_response({"state": "error", "detail": str(st.get("messages"))[-600:]})
        imgs = []
        for _nid, o in (e.get("outputs") or {}).items():
            for im in o.get("images", []):
                imgs.append({"filename": im.get("filename"), "subfolder": im.get("subfolder", ""),
                             "type": im.get("type", "output")})
        return web.json_response({"state": "done", "images": imgs,
                                  "elapsed": round(_time.time() - info.get("t0", _time.time()), 1)})
except Exception:  # 没装 aiohttp 路由也不致命，节点照常可用
    print("[comfy_node_crypt] 手机端路由注册失败:", flush=True)
    _tb.print_exc()
'''

CRYPT_NODE_SRC = CRYPT_NODE_SRC_TEMPLATE  # 安装时用 ALGO 填充，见 install_crypto_node



# ---------- 会话内自续跑：Kaggle 单次上限 12h，到点前主动交班 ----------
# 不续跑的话，手机端在会话被杀那一刻只会看到「连接断开」，没有任何办法自己救回来。
GATE = os.path.join(WORK, "qwen_handoff.json")


def _gate_state():
    """读取交班门闩：auto = 本会话正常存活；handoff = 已被下一个会话接管。"""
    try:
        with open(GATE) as f:
            return json.load(f)
    except Exception:
        return {"auto": True, "note": "no-gate"}


def _set_gate(**kw):
    st = _gate_state()
    st.update(kw)
    st.setdefault("auto", True)
    tmp = GATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, GATE)


def guard_alive():
    """看门狗：隧道/ComfyUI 挂了就重启，别让手机会话悄悄变砖。"""
    try:
        with urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=5) as r:
            r.read()
        return True
    except Exception:
        mark("watchdog: ComfyUI 没响应，重启")
        publish("stage", "restart-comfy")
        start_comfy()
        return False


def wait_for_handoff(session_limit=11.5 * 3600, ticks=int(os.environ.get("QWEN21_TICKS", "0")) or None):
    """
    常驻循环：正常 sleep；一旦门闩被别人翻成 handoff，本会话优雅退出。
    先存 working 让新会话拿到接力棒，再自停，旧容器不留着占额度。
    """
    t0 = time.time()
    n = 0
    last_beat = 0
    while True:
        time.sleep(20)
        n += 1
        if n % 15 == 0:
            guard_alive()
        # 心跳：每 5 分钟把地址再打一行。手机端第一次点按钮时隧道可能刚起来还没扫描到，
        # 有这条重播就不用让人干等到下一次冷启动。
        if time.time() - last_beat > 300:
            last_beat = time.time()
            try:
                publish("ready", "服务在线", url=SERVE_URL[0], gpu=GPU_NAME_GLOBAL[0],
                        egress=SERVE_IP[0], rev=REV, beat=int(time.time()))
            except Exception:
                pass
        st = _gate_state()
        if not st.get("auto", True):
            mark(f"handoff: 已被 {st.get('by', '?')} 接管，本会话退出")
            return "handoff"
        el = time.time() - t0
        if el > session_limit:
            mark("handoff: 接近会话上限，主动交班")
            save_version()
            stop_self()
            return "limit"
        if ticks and n >= ticks:
            mark("handoff: 测试用 tick 上限到达")
            return "ticks"


def save_version():
    """把本会话快照成新版本，让下次内核能读到最新的 URL/令牌/状态。"""
    try:
        from kaggle_secrets import UserSecretsClient  # noqa: F401
        from kaggle.api.kaggle_api_extended import KaggleApi
        api_ = KaggleApi()
        api_.authenticate()
        api_.kernels_output(os.environ.get("KAGGLE_KERNEL_SLUG") or KERNEL_SLUG, path=None, force=True,
                            quiet=True) if False else None
        api_.kernels_status(KERNEL_SLUG)
        api_.kernels_push()
        mark("handoff: 已 push 新版本（接力棒落地）")
        return True
    except Exception as e:
        mark(f"handoff: push 失败 {e}")
        return False


def stop_self():
    """自停：Kaggle 交互会话里有 stop 接口，静默失败也无所谓（等它自然超时）。"""
    try:
        import requests
        requests.post("https://www.kaggle.com/api/i/kaggle.KernelsService/StopKernelSession",
                      json={"kernelId": int(os.environ.get("KAGGLE_KERNEL_RUN_ID") or 0)},
                      timeout=10)
        mark("handoff: 已请求停止当前会话")
    except Exception as e:
        mark(f"handoff: 自停跳过 {e}")


# ---------- 手机端无感拉取：把隧道地址/令牌/状态写进内核日志 ----------
# Kaggle 没有「读某个 kernel 最新版本输出文件」的公开 API，但 CLI 能看到日志。
# 所以日志本身就是通道：每次变化打一行 QWEN21://<base64url(json)>。
KERNEL_SLUG = os.environ.get("KAGGLE_KERNEL_SLUG") or "iceworker/qwen-image-2-1-uncensored-int8-serve"
_TOKEN = None
_LAST = None


def make_token(seed):
    """APK 需要携带的鉴权令牌：seed 的指纹，不含 seed 本身。"""
    return "QW21-" + hashlib.sha256(("qwen-image-2.1|" + seed).encode()).hexdigest()[:32]


def _publish(payload):
    global _LAST, _TOKEN
    if _TOKEN is None:
        _TOKEN = make_token(SEED_GLOBAL)
    payload = dict(payload)
    payload.setdefault("token", _TOKEN)
    payload.setdefault("slug", KERNEL_SLUG)
    payload.setdefault("ts", int(time.time()))
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    line = "QWEN21://" + base64.urlsafe_b64encode(raw).decode().rstrip("=")
    if line == _LAST:
        return
    _LAST = line
    print(line, flush=True)
    # 同时在 working 里留一份，自续跑时新会话能读到上一次的进度
    try:
        with open(os.path.join(WORK, "hub.json"), "w") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


SEED_GLOBAL = "qwen21unc2026"


def publish(kind, msg="", **extra):
    payload = {"kind": kind, "stage": msg or STAGE_GLOBAL[0]}
    payload.update(extra)
    try:
        _publish(payload)
    except Exception:
        pass


STAGE_GLOBAL = ["init"]


def hub_ok():
    """公网可打通的判定：隧道两侧都要通。"""
    try:
        with urllib.request.urlopen(COMFY_URL + "/qwen/health", timeout=10) as r:
            return json.loads(r.read()).get("ok") is True
    except Exception:
        return False


def start_tunnel(port=8188):
    """
    公网入口。Quick Tunnel 每次重启换域名，所以地址必须由云端主动推给手机，
    手机不应该、也不需要知道它会变成什么。
    有 NGROK_TOKEN 时优先用固定域名隧道——域名不变，手机端一次配置永久可用。
    """
    url = None
    ng = os.environ.get("NGROK_TOKEN")
    if ng:
        try:
            sh(f"wget -q -O {TMP}/ngrok.tgz https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz",
               log=os.path.join(LOG_DIR, "tunnel.log"))
            sh(f"tar xzf {TMP}/ngrok.tgz -C {TMP}", log=os.path.join(LOG_DIR, "tunnel.log"))
            sh(f"{TMP}/ngrok config add-authtoken {ng}", log=os.path.join(LOG_DIR, "tunnel.log"))
            sh(f"nohup {TMP}/ngrok http {port} --log=stdout > {LOG_DIR}/ngrok.log 2>&1 &")
            for _ in range(40):
                time.sleep(2)
                m = re.search(r"https://[a-z0-9-]+\.ngrok-free\.(app|dev)",
                              tail_log(os.path.join(LOG_DIR, "ngrok.log"), 8000))
                if m:
                    url = m.group(0)
                    break
        except Exception as e:
            print("ngrok 失败，回落到 quick tunnel:", e, flush=True)

    if not url:
        sh(f"wget -q -O {TMP}/cloudflared "
           f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
           log=os.path.join(LOG_DIR, "tunnel.log"))
        sh(f"chmod +x {TMP}/cloudflared")
        sh(f"nohup {TMP}/cloudflared tunnel --url http://127.0.0.1:{port} --no-autoupdate "
           f"> {LOG_DIR}/tunnel.log 2>&1 &")
        for _ in range(60):
            time.sleep(3)
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",
                          tail_log(os.path.join(LOG_DIR, "tunnel.log"), 8000))
            if m:
                url = m.group(0)
                break

    if not url:
        publish("error", "隧道没起来", detail=tail_log(os.path.join(LOG_DIR, "tunnel.log"), 2000))
        raise RuntimeError("tunnel failed, see logs/tunnel.log")
    # 顺手把"隧道出口 IP 变了"这件事告诉端上：trycloudflare 会轮换出口，
    # 而 ComfyUI 上线时会把当时所有出口 IP 冻成白名单，轮换之后新 IP 就吃 403。
    # 端上拿这条提示清掉旧地址、重新握手，比让用户对着"鉴权没过"猜要靠谱。
    try:
        exit_ip = urllib.request.urlopen("https://api.ipify.org", timeout=8).read().decode().strip()
    except Exception:
        exit_ip = ""
    SERVE_URL[0] = url
    SERVE_IP[0] = exit_ip
    publish("ready", "服务在线", url=url, gpu=GPU_NAME_GLOBAL[0], egress=exit_ip, rev=REV)
    mark(f"serve: 公网地址 {url}")
    print(f"QWEN21 地址：{url}", flush=True)
    return url



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


def probe_image(prefix="ready", steps=6):
    """
    真正跑一次采样，确认"出图"这条腿是通的，而不是只通了隧道和节点路由。

    为什么值得花这 30 秒：int8 权重 + 某个 ComfyUI 版本对不上时，解码出来是
    纯黑或全噪点，接口照样回"完成"，手机照样显示一张图。与其让人盯着黑图猜，
    不如开机就自己跑一张，把亮度和像素方差写进日志。
    """
    w, h = 512, 512
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP_NAME, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["2", 0], "prompt": "a red apple on a wooden table, soft daylight",
                         "negative_prompt": "", "resolution": 1024}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0],
            "seed": 7, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }
    r = run_wf(wf, timeout=1800, want_files=True)
    if r.get("error"):
        publish("error", "probe-image", detail=str(r["error"])[:600])
        print("出图自检失败:", r["error"], flush=True)
        return False
    stat = {"shot": os.path.basename(r["images"][0].get("saved", "")) if r["images"] else "",
            "elapsed": r.get("elapsed")}
    try:
        from PIL import Image as _I
        import numpy as _np
        arr = _np.asarray(_I.open(r["images"][0]["saved"]).convert("RGB")).astype("float32")
        stat.update(bright=round(float(arr.mean()), 1), std=round(float(arr.std()), 1))
        print("出图自检:", stat, flush=True)
        if arr.std() < 2.0:
            # 不 raise：图还是能出，只是可能不好看。把判断交给用户，别把整台机器判死。
            publish("stage", "probe-image", note="图偏黑/纯色，INT8 权重可能与当前 ComfyUI 不匹配")
            return False
    except Exception as e:
        print("出图自检统计失败:", e, flush=True)
    publish("stage", "probe-image", note=json.dumps(stat, ensure_ascii=False))
    return True


# 设备令牌。REV 是写死的稳定值（APK 内置同一份），make_token() 是 seed 派生值。
# 两个都认：换了云端代码而手机没换包时，旧包手上那个也能进。
REV = 7
CLOUD_TOKENS = {"QW21-REV%d" % REV, make_token(SEED_GLOBAL)}


def emit():
    """立刻把当前 stage 打进日志。

    端上只认 QWEN21:// 那几行。第一次举手要等第一个 mark 才发生，
    中间的 pkg 探测、GPU 查询全是静默的——手机上就是「等待云端日志」。
    这里把开局这一行提前发出去：不是给机器看，是给人看「云朵收到了」。
    """
    try:
        publish("stage", STAGE_GLOBAL[0])
    except Exception:
        pass


# 令牌落盘处：self-resume 时读回上一次的进度（只在卷上，不对外暴露）
TOKEN_FILE = "/kaggle/working/qwen_token.txt"


def write_token_file():
    """把当前设备令牌写到 ComfyUI 的 web 根：隧道 /qwen_token.txt 就能读到。

    作用只有一个：设备令牌跟着会话换（交班/重启），而手机上还存着旧的，
    于是云端回 401，用户看到"鉴权没过"却什么都没填错。有了这个文件，
    手机在"隧道通、令牌旧"的情况下能自己把令牌换成现时的，不必重装 App。
    令牌本来就是端与云共享的会话凭据，隧道里再走一遍不算新泄露。
    """
    try:
        web = os.path.join(COMFY, "web")
        os.makedirs(web, exist_ok=True)
        # 一行一个，手机全拿去挨个试；顺序无所谓
        text = "\n".join(sorted(CLOUD_TOKENS)) + "\n"
        for path in (os.path.join(web, "qwen_token.txt"), TOKEN_FILE):
            with open(path, "w") as f:
                f.write(text)
        print("/qwen_token.txt 已就位:", sorted(CLOUD_TOKENS), flush=True)
    except Exception as e:
        print("写令牌文件失败（不影响出图）:", e, flush=True)
