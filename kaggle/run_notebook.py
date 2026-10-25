# ============================================================
#  Qwen-Image-2.1 Uncensored (INT8) · Kaggle 免费 GPU
#  Mode = "run"    只出图（验证用）
#  Mode = "serve"  起常驻服务 + 公网隧道（给 APK 用）
# ============================================================
import json, os, traceback

SEED    = "qwen21unc2026"     # 与 APK 内置一致
Mode    = "run"
PROMPT  = "一只穿西装的猫，赛博朋克城市夜景，电影感打光，85mm 浅景深"
NEG     = ""
STEPS   = 20
W, H    = 1024, 1024
CFG     = 1.0
SEEDNUM = 20261025

STAGE = "init"

def stage(name):
    global STAGE
    STAGE = name
    print(f"\n===== [{name}] =====", flush=True)
    mark(name)


try:
    stage("gpu")
    print("GPU:", has_gpu(), flush=True)

    stage("install")
    install()

    stage("download")
    download_models()

    stage("crypto-node")
    install_crypto_node(SEED)

    stage("start-comfy")
    assert start_comfy(), "ComfyUI 启动失败"

    stage("check-nodes")
    oi = api("/object_info")
    print("EncryptText:", "EncryptText" in oi, "| DecryptText:", "DecryptText" in oi, flush=True)
    print("UNETLoader:", "UNETLoader" in oi, "| TextEncodeQwenImage21:", "TextEncodeQwenImage21" in oi, flush=True)
    stats()

    stage("verify-crypto")
    res_v = verify_crypto(SEED)
    # 密文图落盘后，用同一套 _slot 公式从像素读回码字、独立还原明文：
    # 证明「图本身就是唯一载体」，客户端只需 seed 就能解。
    cipher = next((i["saved"] for i in res_v.get("images", []) if "cipher" in (i.get("filename") or "")), None)
    if cipher:
        from PIL import Image as _Img
        arr = np.array(_Img.open(cipher).convert("RGB"))
        words = [int(v) for v in unrender(arr, len(encode_words(SEED, VERIFY_PROMPT)))]
        back = decode_words(words, SEED)
        print(f"密文图 {arr.shape[1]}x{arr.shape[0]} -> 读回 {len(words)} 码字 -> {back!r}", flush=True)
        assert back == VERIFY_PROMPT, "密文图还原失败"
    print("加密节点闭环 OK", flush=True)

    stage("upload-probe")
    upload_probe("probe")

    stage("generate")
    wf = build_wf(PROMPT, seed=SEEDNUM, steps=STEPS, width=W, height=H, cfg=CFG,
                  enc_text=PROMPT, enc_seed=SEED)
    res = run_wf(wf, timeout=2400)
    if res.get("error"):
        raise RuntimeError("工作流报错: " + str(res["error"])[:2000])
    print("耗时", res.get("elapsed"), "秒", flush=True)
    print("产物", json.dumps(res["images"], ensure_ascii=False), flush=True)

    summary = {
        "prompt": PROMPT, "steps": STEPS, "size": [W, H], "seed": SEEDNUM,
        "elapsed_sec": res["elapsed"],
        "outputs": [{"filename": i.get("filename"), "saved": i.get("saved")} for i in res["images"]],
    }
    open(os.path.join(OUT, "run_summary.json"), "w").write(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if Mode in ("serve", "auto"):
        stage("tunnel")
        import re, threading
        def tunnel():
            sh(f"wget -q -O {WORK}/cloudflared "
               f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64")
            sh(f"chmod +x {WORK}/cloudflared")
            sh(f"nohup {WORK}/cloudflared tunnel --url http://127.0.0.1:8188 --no-autoupdate "
               f"> {LOG_DIR}/tunnel.log 2>&1 &")
            for _ in range(60):
                time.sleep(3)
                t = tail_log(os.path.join(LOG_DIR, "tunnel.log"), 4000)
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", t)
                if m:
                    print("\n>>> PUBLIC URL:", m.group(0), flush=True)
                    open(os.path.join(WORK, "PUBLIC_URL.txt"), "w").write(m.group(0))
                    return
            print("隧道未就绪，查 logs/tunnel.log", flush=True)
        threading.Thread(target=tunnel, daemon=True).start()
        print("=== 服务常驻中，把 PUBLIC URL 填进 APK ===", flush=True)

    print("\nALL DONE", flush=True)

except Exception as e:
    print(f"\n!!! 阶段 [{STAGE}] 失败: {e}", flush=True)
    traceback.print_exc()
    if STAGE in ("start-comfy", "check-nodes", "verify-crypto", "upload-probe", "generate"):
        print("\n---- comfy.log 尾部 ----", flush=True)
        print(tail_log(os.path.join(LOG_DIR, "comfy.log")), flush=True)
    else:
        print("\n---- setup/pip.log 尾部 ----", flush=True)
        print(tail_log(os.path.join(LOG_DIR, "pip.log")), flush=True)
    raise
