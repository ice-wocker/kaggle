# ============================================================
#  Qwen-Image-2.1 Uncensored (INT8) · Kaggle 免费 GPU 一键服务
#  Mode = "run"    只出图（本次默认，验证用）
#  Mode = "serve"  起常驻服务 + 公网隧道（给 APK 用）
# ============================================================
import json, os

SEED    = "qwen21unc2026"     # 与 APK 内置一致
Mode    = "run"
PROMPT  = "一只穿西装的猫，赛博朋克城市夜景，电影感打光，85mm 浅景深"
NEG     = ""
STEPS   = 20
W, H    = 1024, 1024
CFG     = 1.0
SEEDNUM = 20261025

print("GPU:", has_gpu(), flush=True)
install()
download_models()
install_crypto_node(SEED)
assert start_comfy(), "ComfyUI 启动失败"

oi = api("/object_info")
print("EncryptText:", "EncryptText" in oi, "| DecryptText:", "DecryptText" in oi, flush=True)
stats()

# ---- 自检 1：加密节点闭环 ----
verify_crypto(SEED)
print("加密节点闭环 OK", flush=True)

# ---- 自检 2：上传/读取链路 ----
upload_probe("probe")

# ---- 正式出图 ----
print("\n=== 开始出图 ===", flush=True)
wf = build_wf(PROMPT, seed=SEEDNUM, steps=STEPS, width=W, height=H, cfg=CFG,
              enc_text=PROMPT, enc_seed=SEED)
res = run_wf(wf, timeout=2400)
print("耗时", res["elapsed"], "秒", flush=True)
print("产物", json.dumps(res["images"], ensure_ascii=False), flush=True)

# 汇总，留给 APK / 人工核验
summary = {"prompt": PROMPT, "steps": STEPS, "size": [W, H], "seed": SEEDNUM,
           "elapsed_sec": res["elapsed"],
           "outputs": [{"filename": i.get("filename"), "saved": i.get("saved")} for i in res["images"]]}
open(os.path.join(OUT, "run_summary.json"), "w").write(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
