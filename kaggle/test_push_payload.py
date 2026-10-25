# -*- coding: utf-8 -*-
"""校验 push 载荷是否符合 Kaggle 真正的 ApiSaveKernelRequest 契约。

为什么要有这个文件：那句 `400 A language must be specified.` 让人改了三轮语言字段，
而 language 一直都是 python。真相是载荷结构不对——我们发的是 {source:{...}} 的嵌套壳，
官方发的是扁平对象。服务端读不懂 source，就拿"下一条必填项"language 来报错。
所以这里把契约固化成断言：结构错了本地就红，不用再等一次线上 400。

用法：
  python3 kaggle/test_push_payload.py            # 只校验结构（离线，随时可跑）
  python3 kaggle/test_push_payload.py --key KGAT_xxx   # 再拿真令牌打一次真实端点
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JAVA = os.path.join(ROOT, "android", "app", "src", "com", "cb", "qwen21", "Kaggle.java")

# Kaggle 官方 kagglesdk 里 ApiSaveKernelRequest 的全部字段（camelCase 序列化名）。
# 只有这些键会被服务端读进去，多一个不认识的顶层键都会让它把请求当成不完整。
ALLOWED = {
    "id", "slug", "newTitle", "text", "language", "kernelType",
    "datasetDataSources", "kernelDataSources", "competitionDataSources",
    "categoryIds", "isPrivate", "enableGpu", "enableInternet",
    "dockerImagePinningType", "modelDataSources", "enableTpu",
    "sessionTimeoutSeconds", "priority", "dockerImage",
    "kernelExecutionType", "machineShape",
}
REQUIRED = {"slug", "text", "language", "kernelType"}


def java_source():
    with open(JAVA, encoding="utf-8") as f:
        return f.read()


def payload_shape(src):
    """从 Kaggle.java 里摘出 push 到底 put 了哪些键。

    不跑 Java 也能看：push 里每一个 s.put("k", ...) 就是它会发出去的键。
    这条断言拦的是"又有人手滑加了一层嵌套壳"。
    """
    i = src.find("String push(String owner, String apiKey, Progress p)")
    if i < 0:
        raise SystemExit("找不到 push(owner, apiKey, Progress)，先确认源码结构")
    body = src[i:i + 6000]
    keys = []
    marker = 's.put("'
    for part in body.split(marker)[1:]:
        keys.append(part.split('"', 1)[0])
    return body, keys


def check_offline():
    src = java_source()
    body, keys = payload_shape(src)
    bad = []
    if not keys:
        bad.append("push 里一个 s.put 都没找到，解析失败")
    extra = [k for k in keys if k not in ALLOWED]
    if extra:
        bad.append("载荷里有 Kaggle 不认识的顶层键 %s——服务端读不懂就报 language，"
                   "正是那句 400 的来源" % extra)
    missing = [k for k in REQUIRED if k not in keys]
    if missing:
        bad.append("载荷缺必填 %s" % missing)
    if "source" in keys:
        bad.append("又套了 source 外壳：官方契约是扁平对象，没有 source")
    # 只看真正 put 出去的键：注释里提到 sessionEnv 不算
    if "sessionEnv" in keys:
        bad.append("sessionEnv 不是 ApiSaveKernelRequest 的字段，云端拿不到它；"
                   "令牌要走 KAGGLE_API_TOKEN / kaggle_secrets 那条路")
    if 'owner = "me"' in body:
        bad.append("push 里还在拿 me 兜账号")
    if '"language", "python"' not in body:
        bad.append("顶层 language 没写死 python")
    if 'inner.put(' in body:
        bad.append("还在拼内层 slug 对象：官方契约没有它")
    # 端点：两个都要能打，主用 midtier 的 RPC
    if "kernels.KernelsApiService/SaveKernel" not in src:
        bad.append("没打 midtier 的 SaveKernel 端点")
    if "api/v1/kernels/push" not in src:
        bad.append("没留 www 旧端点的兼容路径")
    return bad, keys


def sample_payload(owner="someone"):
    """和 Kaggle.java 拼出来的那份同形：扁平、字段名 camelCase。"""
    return {
        "slug": "%s/qwen-image-2-1-uncensored-int8-serve" % owner,
        "newTitle": "Qwen Image 2.1 Uncensored INT8 Serve",
        "text": json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}),
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": True,
        "datasetDataSources": [],
        "kernelDataSources": [],
        "competitionDataSources": [],
        "categoryIds": [],
    }


def check_online(key, owner):
    """拿真令牌打一次，只看「是不是那句 language」——那是结构被拒的指纹。"""
    import urllib.error
    import urllib.request

    body = json.dumps(sample_payload(owner)).encode("utf-8")
    endpoints = [
        "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel",
        "https://www.kaggle.com/api/v1/kernels/push",
    ]
    failures = []
    for url in endpoints:
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + key)
        req.add_header("User-Agent", "Qwen21APK/2.0")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                code, text = r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            code, text = e.code, e.read().decode("utf-8", "replace")
        except Exception as e:  # 网络问题不是契约问题，单列出来
            print("  %s -> 连不上 %s" % (url, e))
            continue
        low = text.lower()
        if "language must be specified" in low or "must be specified" in low:
            failures.append("%s 仍被结构性地拒了：HTTP %d %s" % (url, code, text[:200]))
            continue
        # 200 说明真推上去了；400/401/403 只要不是那句 language，就说明载荷已经读进去了
        print("  %s -> HTTP %d %s" % (url, code, text[:160].replace("\n", " ")))
        if code >= 500:
            failures.append("%s 服务端错误 HTTP %d" % (url, code))
    return failures


def check_sdk():
    """和官方 kagglesdk 的 ApiSaveKernelRequest 逐字段对齐（装了才跑）。

    这是最硬的一版断言：不是「我觉得这些键对」，是官方类自己说哪些键合法。
    """
    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    except Exception:
        print("跳过 SDK 对比：没装 kagglesdk（pip install kaggle）")
        return []

    def camel(s):
        p = s.split("_")
        return p[0] + "".join(x[:1].upper() + x[1:] for x in p[1:])

    legal = {camel(f[1:]) for f in ApiSaveKernelRequest().__dict__
             if f.startswith("_") and f != "_is_frozen"}
    _, keys = payload_shape(java_source())
    illegal = [k for k in keys if k not in legal]
    if illegal:
        return ["载荷里有官方契约不认识的键 %s" % illegal]
    print("OK 与官方 ApiSaveKernelRequest 对齐（%d 个键全在官方的 %d 个合法字段里）"
          % (len(keys), len(legal)))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("KAGGLE_API_TOKEN", ""))
    ap.add_argument("--owner", default=os.environ.get("KAGGLE_OWNER", ""))
    args = ap.parse_args()

    bad, keys = check_offline()
    print("载荷顶层键：%s" % ", ".join(keys))
    if bad:
        for b in bad:
            print("FAIL " + b)
        return 1
    print("OK 结构符合 ApiSaveKernelRequest（扁平、无 source、无 sessionEnv）")
    sdk_bad = check_sdk()
    if sdk_bad:
        for b in sdk_bad:
            print("FAIL " + b)
        return 1

    if not args.key:
        print("跳过联网：没给 --key")
        return 0
    if not args.owner:
        print("跳过联网：没给 --owner（真 handle，不能是 me）")
        return 0
    print("联网实测 · 推给 %s：" % args.owner)
    fails = check_online(args.key, args.owner)
    for f in fails:
        print("FAIL " + f)
    if fails:
        return 1
    print("OK 两个端点都没再吐 language must be specified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
