# -*- coding: utf-8 -*-
"""一次出 9 张，按 run_summary.json 那次的口径计时。

为什么不是「在 notebook 里写个 for 循环」：那样拿到的是云端自己报的 elapsed，
跟 README 里 236 秒不是一回事——那 236 秒是端到端用户在等的墙钟时间。
所以这里走跟 APK 完全一样的路：信封封包 -> POST /qwen/generate -> 轮询 /qwen/task
-> 取回图片，每一秒都算在账上。

用法：
  python3 kaggle/gen_batch.py --url https://xxx.trycloudflare.com --token QW21-…
  python3 kaggle/gen_batch.py --wait --token QW21-…   # 只看得到 slug 时，自己从内核日志里捞地址
  python3 kaggle/gen_batch.py --dry-run               # 不出网：只封包 + 打计划

产物：docs/bench/q01.png … q09.png（--out 可改）+ docs/bench/batch.json（逐张耗时）
"""
import argparse, base64, collections, hashlib, json, os, struct, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEED = "qwen21unc2026"          # 与 APK / 云端一致

# 9 张铺开在「同一提示词、不同种子」与「同种子、不同题材」两轴之间：
# 只有固定提示词，236 秒的方差就全算进测量里；全是题材差异，又看不出单张成本。
PROMPT = "一只穿西装的猫，赛博朋克城市夜景，电影感打光，85mm 浅景深"
SEEDS = [20261025, 20261026, 20261027,          # 同一句提示词，换种子看稳定性
         7, 42, 2026,                            # 同种子，换题材看耗时是否跟内容相关
         314159, 271828, 141421]


def key(seed=SEED.encode()):
    return hashlib.sha256(seed).digest()


M32 = 0xFFFFFFFF


def rotl(x, n):
    x &= M32
    return ((x << n) | (x >> (32 - n))) & M32


def keystream(k, tag, n):
    out, ctr, t = b"", 0, tag.encode()
    while len(out) < n:
        out += hashlib.sha256(k + t + struct.pack("<I", ctr)).digest()
        ctr += 1
    return out[:n]


def subkeys(k, rounds):
    return [struct.unpack(">I", hashlib.sha256(k + b"k" + bytes([48 + i])).digest()[:4])[0]
            for i in range(rounds)]


def _i2b(v):
    return struct.unpack(">i", struct.pack(">I", v & M32))[0]


def encrypt(plain, k):
    pad = 8 - (len(plain) % 8)
    data = plain + bytes([pad]) * pad
    sk, out = subkeys(k, 4), b""
    for off in range(0, len(data), 8):
        l, r = struct.unpack(">II", data[off:off + 8])
        for kk in sk:
            l, r = r, _i2b(l ^ ((rotl(r, 3) + kk) & M32))
        out += struct.pack(">II", l & M32, r & M32)
    ks = keystream(k, "diffuse", len(out))
    return bytes(a ^ b for a, b in zip(out, ks))


def _field(kn, v):
    kb = kn.encode()
    if isinstance(v, bool):
        typ, raw = 0, bytes([1 if v else 0])
    elif isinstance(v, int):
        typ = 1 if v >= 0 else 2
        raw = abs(v).to_bytes(8, "big")
    elif isinstance(v, float):
        typ, raw = 3, struct.pack(">d", v)
    else:
        typ, raw = 4, str(v).encode()
    return bytes([len(kb)]) + kb + struct.pack(">H", len(raw)) + bytes([typ]) + raw


def pack_task(fields, k):
    inner = b"Q2" + b"".join(_field(a, b) for a, b in fields.items())
    ct = encrypt(inner, k)
    return b"\xcb\x22\x01\x00" + struct.pack(">H", len(ct)) + ct


def envelope(prompt, seed, steps, w, h, cfg=1.0, batch=1, negative="", nsfw=True, k=None):
    f = collections.OrderedDict([
        ("prompt", prompt), ("negative", negative), ("steps", steps), ("cfg", float(cfg)),
        ("seed", int(seed)), ("batch", int(batch)), ("width", int(w)), ("height", int(h)),
        ("nsfw", bool(nsfw)),
    ])
    return pack_task(f, k or key())


def _post(url, payload, token, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "X-QWEN-TOKEN": token})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url, token, timeout=60):
    req = urllib.request.Request(url, headers={"X-QWEN-TOKEN": token})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _view(base, token, im):
    q = "?" + urllib.parse.urlencode({"filename": im["filename"],
                                      "subfolder": im.get("subfolder", ""),
                                      "type": im.get("type", "output")})
    req = urllib.request.Request(base.rstrip("/") + "/view" + q, headers={"X-QWEN-TOKEN": token})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def one(base, token, prompt, seed, steps, w, h, cfg, out_png, timeout=900):
    """跑一张，返回墙钟。和手机端一样：计时从封包前开始，到图片落盘为止。"""
    t0 = time.time()
    env = envelope(prompt, seed, steps, w, h, cfg)
    r = _post(base.rstrip("/") + "/qwen/generate",
              {"t": base64.b64encode(env).decode(), "k": token}, token)
    if "id" not in r:
        raise RuntimeError("提交失败: %s" % r)
    tid = r["id"]
    while time.time() - t0 < timeout:
        t = _get(base.rstrip("/") + "/qwen/task/" + tid, token)
        st = t.get("state")
        if st == "done":
            imgs = t.get("images") or []
            if not imgs:
                raise RuntimeError("任务完成但没有图")
            data = _view(base, token, imgs[0])
            open(out_png, "wb").write(data)
            return round(time.time() - t0, 1), len(data)
        if st == "error":
            raise RuntimeError("生成失败: %s" % str(t.get("detail"))[:400])
        time.sleep(1.5)
    raise RuntimeError("超时 %ds" % timeout)


# ---------- 只看得到 slug 时：自己去内核日志里捞地址 ----------
# 手机之所以能拿到地址，是因为它读得到日志。人在命令行上其实也一样，
# 只是要自己把 QWEN21:// 那行解出来——这跟「用浏览器翻日志」是同一件事，
# 不属于任何特殊权限。用你自己的 KGAT_ 令牌，读你自己名下的内核。
KAGGLE_LOG = "https://www.kaggle.com/api/v1/kernels/output"


def hint_slug():
    """没给 slug 时，至少在报错里说清楚该填什么。"""
    m = os.environ.get("KAGGLE_KERNEL_SLUG")
    if m:
        return m
    raise SystemExit("--wait 需要 --slug owner/kernel-slug（跟你 push 时 payload 里那个一致）")


def read_log(slug, token, tries=1, gap=20, wait_ready=True):
    """拉内核日志，返回 (base_url, token)。没有就等。"""
    key = token or os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY") or ""
    if not key:
        raise SystemExit("--wait 需要 Kaggle 凭据：--key KGAT_… 或环境变量 KAGGLE_API_TOKEN")
    owner = slug.split("/")[0]
    for i in range(tries):
        q = urllib.parse.urlencode({"owner": owner, "kernel": slug})
        req = urllib.request.Request(KAGGLE_LOG + "?" + q,
                                     headers={"Authorization": "Bearer " + key,
                                              "User-Agent": "qwen21-gen-batch"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception as e:
            print("读日志失败（第 %d 次）：%s" % (i + 1, e), flush=True)
            text = ""
        found = None
        for line in reversed(text.splitlines()):
            at = line.find("QWEN21://")
            if at < 0:
                continue
            b = line[at + len("QWEN21://"):].strip()
            try:
                b += "=" * (-len(b) % 4)
                found = json.loads(base64.urlsafe_b64decode(b).decode("utf-8"))
                break
            except Exception:
                continue
        if found and found.get("kind") == "ready" and found.get("url"):
            print("日志里拿到地址：%s" % found["url"], flush=True)
            return found["url"], found.get("token") or ""
        if found:
            print("日志最新一条是 %r，还没 ready" % found.get("kind"), flush=True)
        if wait_ready and i + 1 < tries:
            time.sleep(gap)
    raise SystemExit("等了 %d 次也没在日志里看到 ready。首次要装环境+拉 17GB 权重，10-15 分钟属正常。"
                     % tries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("QWEN21_URL", ""))
    ap.add_argument("--token", default=os.environ.get("QWEN21_TOKEN", ""))
    ap.add_argument("--wait", action="store_true", help="没有 --url 时，从内核日志里等 ready")
    ap.add_argument("--slug", default=os.environ.get("KAGGLE_KERNEL_SLUG", ""))
    ap.add_argument("--key", default="", help="读日志用的 Kaggle 令牌（KGAT_…）")
    ap.add_argument("--tries", type=int, default=60, help="--wait 轮数，默认 60×20s=20 分钟")
    ap.add_argument("--gap", type=int, default=20)
    ap.add_argument("--count", type=int, default=9)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "bench"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    themes = [PROMPT, "雨夜的霓虹拉面摊，蒸汽升腾，胶片质感",
              "雪山日出，广角，冷调，长曝光云层", "老式打字机特写，暖光台灯，浅景深",
              "机械蜂鸟，黄铜与玻璃，微距", "江南水乡清晨，薄雾，水墨感",
              "宇航员在麦田里摘西瓜，晴天", "玻璃缸里的小型雷暴，微距，暗背景"]
    plan = []
    for i in range(a.count):
        plan.append({
            "n": i + 1,
            "seed": SEEDS[i % len(SEEDS)] + (i // len(SEEDS)),
            "prompt": PROMPT if i % 3 == 0 else themes[i % len(themes)],
            "file": "q%02d.png" % (i + 1),
        })

    if a.dry_run:
        print("dry-run：只封包，不出网")
        for p in plan:
            env = envelope(p["prompt"], p["seed"], a.steps, a.size, a.size, a.cfg)
            print("  q%02d  seed=%-9d env=%4dB  %s" % (p["n"], p["seed"], len(env), p["prompt"][:26]))
        return 0

    if not a.url and a.wait:
        slug = a.slug or hint_slug()
        a.url, tok = read_log(slug, a.key, a.tries, a.gap)
        a.token = a.token or tok

    if not a.url or not a.token:
        print("缺 --url / --token：这两个只有云端日志里那行 QWEN21:// 会告诉你。", file=sys.stderr)
        return 2

    h = _get(a.url.rstrip("/") + "/qwen/health", a.token)
    if not h.get("auth"):
        print("令牌不对：/qwen/health 回 %s" % h, file=sys.stderr)
        return 2
    print("健康检查通过，队列 %s，开始 %d 张" % (h.get("queue"), a.count), flush=True)

    rows, t_all = [], time.time()
    for p in plan:
        path = os.path.join(a.out, p["file"])
        try:
            sec, size = one(a.url, a.token, p["prompt"], p["seed"], a.steps, a.size, a.size, a.cfg, path)
            rows.append(dict(p, elapsed_sec=sec, bytes=size, ok=True))
            print("  q%02d %6.1fs %7dB  %s" % (p["n"], sec, size, p["prompt"][:24]), flush=True)
        except Exception as e:
            rows.append(dict(p, error=str(e)[:300], ok=False))
            print("  q%02d 失败：%s" % (p["n"], e), flush=True)

    ok = [r for r in rows if r["ok"]]
    total = round(time.time() - t_all, 1)
    summary = {
        "prompt": PROMPT, "steps": a.steps, "size": [a.size, a.size], "cfg": a.cfg,
        "count": a.count, "ok": len(ok), "total_sec": total,
        "avg_sec": round(sum(r["elapsed_sec"] for r in ok) / len(ok), 1) if ok else None,
        "min_sec": min((r["elapsed_sec"] for r in ok), default=None),
        "max_sec": max((r["elapsed_sec"] for r in ok), default=None),
        "rows": rows,
    }
    with open(os.path.join(a.out, "batch.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n%d/%d 张成功，总墙钟 %.1fs，单张均 %.1fs（%s–%s）"
          % (len(ok), a.count, total, summary["avg_sec"] or 0, summary["min_sec"], summary["max_sec"]))
    return 0 if len(ok) == a.count else 1


if __name__ == "__main__":
    sys.exit(main())
