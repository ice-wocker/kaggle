# ============================================================
#  Qwen-Image-2.1 Uncensored (INT8) · Kaggle 免费 GPU
#  手机端只填一个 Kaggle API Key，剩下的这里全包：
#  自续跑 -> 装环境 -> 拉权重 -> 起 ComfyUI -> 开公网隧道 -> 把地址写进日志
# ============================================================
import json, os, re, time, traceback

SEED = "qwen21unc2026"          # 与 APK 内置一致
Mode = os.environ.get("QWEN21_MODE", "serve")   # serve = 常驻给 APK 用；run = 只出图
PROMPT = "一只穿西装的猫，赛博朋克城市夜景，电影感打光，85mm 浅景深"
STEPS = 20
W, H = 1024, 1024
CFG = 1.0
SEEDNUM = 20261025
LIMIT_HOURS = float(os.environ.get("QWEN21_LIMIT_HOURS", "11.5"))

STAGE = "init"
STAGE_GLOBAL[0] = STAGE


def stage(name):
    global STAGE
    STAGE = name
    STAGE_GLOBAL[0] = name
    print(f"\n===== [{name}] =====", flush=True)
    publish("stage", name)
    mark(name)


try:
    stage("gpu")
    GPU_NAME_GLOBAL[0] = "T4" if has_gpu() else "CPU"
    _set_gate(auto=True, by=os.environ.get("KAGGLE_KERNEL_RUN_ID"))
    publish("stage", "gpu", note=GPU_NAME_GLOBAL[0])

    stage("install")
    install()

    stage("download")
    download_models()

    stage("crypto-node")
    install_crypto_node(SEED)

    stage("start-comfy")
    # 起不来时把 comfy.log 的思路给出来：绝大多数是依赖缺一个硬 import 的包。
    assert start_comfy(), "ComfyUI 启动失败（看下面 comfy.log 尾部，通常是 ModuleNotFoundError）"

    stage("check-nodes")
    oi = api("/object_info")
    need_nodes = ["EncryptText", "DecryptText", "UNETLoader", "CLIPLoader", "TextEncodeQwenImage21", "KSampler"]
    lacking = [n for n in need_nodes if n not in oi]
    print("节点:", {n: (n in oi) for n in need_nodes}, flush=True)
    if lacking:
        raise RuntimeError(f"缺节点 {lacking}，int8 权重可能不匹配这个 ComfyUI 版本")
    stats()

    stage("verify-crypto")
    res_v = verify_crypto(SEED)
    cipher = next((i["saved"] for i in res_v.get("images", []) if "cipher" in (i.get("filename") or "")), None)
    if cipher:
        from PIL import Image as _Img
        arr = np.array(_Img.open(cipher).convert("RGB"))
        words = [int(v) for v in unrender(arr, len(encode_words(SEED, VERIFY_PROMPT)))]
        back = decode_words(words, SEED)
        print(f"密文图 {arr.shape[1]}x{arr.shape[0]} -> 读回 {len(words)} 码字 -> {back!r}", flush=True)
        assert back == VERIFY_PROMPT, "密文图还原失败"
    env = pack_task([("prompt", VERIFY_PROMPT), ("steps", 20), ("cfg", 1.0), ("seed", 7),
                     ("batch", 1), ("width", 1024), ("height", 1024), ("negative", ""),
                     ("nsfw", True)], SEED)
    assert unpack_task(env, SEED)["prompt"] == VERIFY_PROMPT, "任务信封往返失败"
    print("加密闭环 OK（密文图 + 任务信封）", flush=True)

    stage("serve")
    url = start_tunnel(8188)
    print(f"\n>>> APK 无需任何配置，地址与令牌已随日志下发：{url}", flush=True)

    # 手机是后来才连上来的，前面那些 stage 行它一条都没见过。
    # 补一版现状，让人一打开 App 就看到"到哪一步了"，不用猜。
    publish("stage", "serve", note=f"已就绪 · GPU {GPU_NAME_GLOBAL[0]} · 地址已下发")

    # 真跑一张小图：接口通不通、INT8 权重在当前 ComfyUI 上能不能出正常图，
    # 都在这一步给答案。跑完把结果写回日志，手机端点按钮前就知道云端是"能用"还是"能连但画不出来"。
    stage("probe-image")
    ok_probe = probe_image("ready", steps=6)
    stage("serve")
    publish("ready", "服务在线", url=url, gpu=GPU_NAME_GLOBAL[0],
            egress=SERVE_IP[0], rev=REV, probe="ok" if ok_probe else "weak")

    if Mode == "run":
        stage("generate")
        wf = build_wf(PROMPT, seed=SEEDNUM, steps=STEPS, width=W, height=H, cfg=CFG,
                      enc_text=PROMPT, enc_seed=SEED)
        res = run_wf(wf, timeout=2400)
        if res.get("error"):
            raise RuntimeError("工作流报错: " + str(res["error"])[:2000])
        print("耗时", res.get("elapsed"), "秒", flush=True)
        print("\nALL DONE", flush=True)
    else:
        # 常驻：等手机来活；到点或有人接管就交班
        stage("handoff")
        reason = wait_for_handoff(session_limit=LIMIT_HOURS * 3600)
        print(f"会话结束: {reason}", flush=True)

except Exception as e:
    print(f"\n!!! 阶段 [{STAGE}] 失败: {e}", flush=True)
    traceback.print_exc()
    publish("error", STAGE, detail=str(e)[:800])
    if STAGE in ("start-comfy", "check-nodes", "verify-crypto", "generate", "serve", "probe-image"):
        print("\n---- comfy.log 尾部 ----", flush=True)
        print(tail_log(os.path.join(LOG_DIR, "comfy.log")), flush=True)
        print("\n---- tunnel.log 尾部 ----", flush=True)
        print(tail_log(os.path.join(LOG_DIR, "tunnel.log")), flush=True)
    else:
        print("\n---- setup/pip.log 尾部 ----", flush=True)
        print(tail_log(os.path.join(LOG_DIR, "pip.log")), flush=True)
    raise
