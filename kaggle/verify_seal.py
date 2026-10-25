import hashlib, struct

SEED = "qwen21unc2026"
M32 = 0xFFFFFFFF

def key(seed=b"qwen21unc2026"): return hashlib.sha256(seed).digest()

def rotl(x, n):
    x &= M32
    return ((x << n) | (x >> (32 - n))) & M32

def keystream(k, tag, n):
    out=b""; ctr=0; t=tag.encode("ascii")
    while len(out) < n:
        out += hashlib.sha256(k + t + struct.pack("<I", ctr)).digest(); ctr += 1
    return out[:n]

def subkeys(k, rounds):
    sk=[]
    for i in range(rounds):
        d = hashlib.sha256(k + b"k" + bytes([ord('0')+i])).digest()
        sk.append(struct.unpack(">I", d[:4])[0])
    return sk

def i2b(v): return struct.unpack(">i", struct.pack(">I", v & M32))[0]

def encrypt(plain, k):
    pad = 8 - (len(plain) % 8)
    data = plain + bytes([pad])*pad
    sk = subkeys(k, 4)
    out = b""
    for off in range(0, len(data), 8):
        l, r = struct.unpack(">II", data[off:off+8])
        for kk in sk:
            nl = r
            r = i2b((l ^ ((rotl(r,3) + kk) & M32)))
            l = nl
        out += struct.pack(">II", l & M32, r & M32)
    ksd = keystream(k, "diffuse", len(out))
    return bytes(a ^ b for a, b in zip(out, ksd))

def decrypt(cipher, k):
    sk = subkeys(k, 4)
    ksd = keystream(k, "diffuse", len(cipher))
    data = bytes(a ^ b for a, b in zip(cipher, ksd))
    out = b""
    for off in range(0, len(data), 8):
        l, r = struct.unpack(">II", data[off:off+8])
        for kk in reversed(sk):
            nr = l
            l = i2b(r ^ ((rotl(l,3) + kk) & M32))
            r = nr
        out += struct.pack(">II", l & M32, r & M32)
    pad = out[-1]
    assert 1 <= pad <= 8, f"bad padding {pad}"
    assert out[-pad:] == bytes([pad])*pad, "padding mismatch"
    return out[:-pad]

def encodeTask(prompt, k):
    ct = encrypt(prompt.encode(), k)
    return b"\xcb\x21\x01\x00" + struct.pack(">H", len(ct)) + ct

PROMPT = "一只穿西装的猫，赛博朋克城市夜景，电影感打光，85mm 浅景深"
k = key()
blob = encodeTask(PROMPT, k)
print("encodeTask len :", len(blob))
print("hex[:40]       :", blob.hex()[:40])
print("roundtrip ok   :", decrypt(blob[6:], k).decode() == PROMPT)
print("== 与设备屏幕对比 ==")
dev = "cb210100005892cb064e14e381e64030b57ac39c"
print("device  :", dev)
print("python  :", blob.hex()[:40])
print("MATCH   :", blob.hex()[:40] == dev)

# ---- 任务信封 packTask 交叉验证（与设备日志里的 ① 本地封装 对应）----
def field_bytes(kname, v):
    kb = kname.encode()
    if isinstance(v, bool):
        typ, raw = 0, bytes([1 if v else 0])
    elif isinstance(v, int):
        typ = 1 if v >= 0 else 2
        raw = abs(v).to_bytes(8, "big")
    elif isinstance(v, float):
        typ = 3
        raw = struct.pack(">d", v)
    else:
        typ, raw = 4, str(v).encode()
    return bytes([len(kb)]) + kb + struct.pack(">H", len(raw)) + bytes([typ]) + raw

def pack_task(fields, k):
    inner = b"Q2"
    for kk, vv in fields.items():
        inner += field_bytes(kk, vv)
    ct = encrypt(inner, k)
    return b"\xcb\x22\x01\x00" + struct.pack(">H", len(ct)) + ct

import collections
f = collections.OrderedDict([
    ("prompt", PROMPT), ("negative", ""), ("steps", 20), ("cfg", 1.0),
    ("seed", 20261025), ("batch", 1), ("width", 1024), ("height", 1024),
    ("nsfw", False),
])
env = pack_task(f, k)
print()
print("== 信封 packTask ==")
print("len  :", len(env))
print("hex  :", env.hex()[:32], "...")

# 还原验证：解出来字段逐个比对
inner = decrypt(env[6:], k)
assert inner[:2] == b"Q2"
o = 2; got = {}
while o < len(inner):
    kl = inner[o]; o += 1
    kn = inner[o:o+kl].decode(); o += kl
    vl = struct.unpack(">H", inner[o:o+2])[0]; o += 2
    typ = inner[o]; o += 1
    raw = inner[o:o+vl]; o += vl
    if typ == 0: val = raw[0] == 1
    elif typ in (1,2): val = int.from_bytes(raw, "big") * (-1 if typ==2 else 1)
    elif typ == 3: val = struct.unpack(">d", raw)[0]
    else: val = raw.decode()
    got[kn] = val
print("解封字段:", got)
print("字段全等:", dict(got) == dict(f))
