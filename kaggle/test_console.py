# -*- coding: utf-8 -*-
"""控制台/回显那几个口的自检：不碰 ComfyUI、不联网。

云端新加的三件事——/qwen/steps、/qwen/preview、/qwen/console.json——都活在一个
裸 try 里（路由装不上也不能把加密节点带下去），所以它们最容易「改完没人跑」。
这里把「能离线验的那部分」拆出来验：页面能画、快照结构对、解包回显的字段名对得上。
    python3 kaggle/test_console.py
"""
import io, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC = io.open(os.path.join(HERE, "qwen_setup.py"), encoding="utf-8").read()
FAIL = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else "  " + str(detail)))
    if not ok:
        FAIL.append(name)


def _PACK(collections):
    """用端上的字段表封一封真信封（走 verify_seal 的独立实现）。"""
    import verify_seal as vs
    fields = collections.OrderedDict([("prompt", "一只穿西装的猫"), ("steps", 20), ("width", 1024),
                                      ("height", 1024), ("seed", 20261025), ("nsfw", True)])
    return vs.pack_task(fields, vs.key())


def block(start, end):
    i = SRC.index(start)
    j = SRC.index(end, i)
    return SRC[i:j]


# 1) 控制台页面本身是一张自包含的 HTML：断网也得能画出来
html = re.search(r'CONSOLE_HTML = """(.*?)"""', SRC, re.S).group(1)
check("控制台是完整 HTML", html.lstrip().startswith("<!doctype html") and "</html>" in html)
# 只允许出现在提示文案里（教人怎么拼地址），不能是个真令牌
leaked = [l for l in html.splitlines() if "QW21-" in l and "token=" in l and "?" not in l]
check("页面里没有硬编码令牌", not leaked, leaked[:1])
check("页面读 console.json", "/qwen/console.json" in html)
check("页面能拉过程图", "/qwen/preview/" in html)

# 2) 新增的四个路由都在
for route in ('@PromptServer.instance.routes.get("/qwen/console")',
              '@PromptServer.instance.routes.get("/qwen/console.json")',
              '@PromptServer.instance.routes.get("/qwen/steps/{tid}")',
              '@PromptServer.instance.routes.get("/qwen/preview/{tid}")',
              '@PromptServer.instance.routes.post("/qwen/decrypt")',
              '@PromptServer.instance.routes.post("/qwen/generate")'):
    check("路由存在 " + route.split('routes.')[1], route in SRC)

# 3) 每个新口都要鉴权：顺手把令牌当门禁的唯一入口是 _auth
for name in ("_qwen_console_json", "_qwen_steps", "_qwen_preview", "_qwen_decrypt"):
    body = block("async def %s(request):" % name, "@PromptServer.instance.routes")
    check("%s 先验令牌" % name, "_auth(request" in body)

# 4) 控制台页故意不鉴权：它是壳子，数据在 console.json 里
shell = block('async def _qwen_console(request):', "@PromptServer.instance.routes")
check("控制台壳子不含数据", "console_snapshot" not in shell and "TOKENS" not in shell)

# 5) 快照结构：字段名与手机/网页两边读的是同一套
snap = block("def console_snapshot():", "def step_report(")
for key in ('"stage"', '"stages"', '"task"', '"lines"', '"queue"', '"url"', '"rev"'):
    check("快照含 %s" % key, key in snap)

# 6) 回显只给字段名和长度，不给值：密文流出去可以，明文不行
stats = block("def _envelope_stats(blob):", "@PromptServer.instance.routes.get(\"/qwen/console\")")
check("回显只列 k/n", '"k": k' in stats and '"n": vl' in stats)
check("回显不带字段值", "f[k]" not in stats and "out[k]" not in stats)

# 7) 阶段时间线去重：同名阶段只记第一次，不然页面会被 serve 刷满
mark = block("def _remember_stage(name):", "def tail_log(")
check("阶段去重", "STAGE_HISTORY_KEYS" in mark and "if key in STAGE_HISTORY_KEYS" in mark)

# 8) 真的把节点模板装起来跑一遍：路由全在、信封在里面也解得开。
#    只 grep 字串是骗自己的——模板里那段活在裸 try 里，写错了不会有人报错。
def boot_node(src=SRC):
    import base64, types
    algo_start = src.index("MIN_SIDE = 32")
    stops = [src.index(x) for x in ("class EncryptText:", "CONSOLE_HTML =", "CRYPT_NODE_SRC_TEMPLATE =",
                                    "CRYPT_NODE_SRC =") if x in src]
    algo = src[algo_start:min(stops)]
    a = "CRYPT_NODE_SRC_TEMPLATE = r\'\'\'"
    b = "CRYPT_NODE_SRC = CRYPT_NODE_SRC_TEMPLATE"
    t = src[src.index(a) + len(a):src.index(b)]
    tpl = t[:t.rindex("\'\'\'")]
    node = (tpl.replace("@@ALGO@@", algo).replace("@@TOKENS@@", repr(["QW21-TEST"]))
               .replace("@@SEED@@", "qwen21unc2026")
               .replace("@@UNET@@", "u").replace("@@CLIP@@", "c").replace("@@VAE@@", "v"))
    routes = {}

    class Routes:
        def get(self, p): return lambda f: routes.setdefault(("GET", p), f)
        def post(self, p): return lambda f: routes.setdefault(("POST", p), f)

    class PS:
        instance = types.SimpleNamespace(routes=Routes(), prompt_queue=None)

    fake = types.ModuleType("server"); fake.PromptServer = PS
    saved = {k: sys.modules.get(k) for k in ("server", "execution", "torch", "numpy")}
    sys.modules["server"] = fake
    sys.modules["execution"] = types.ModuleType("execution")
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch"); torch.Tensor = object
        sys.modules["torch"] = torch
    try:
        import aiohttp  # noqa: F401  节点里 from aiohttp import web
    except Exception:
        return {}, {}
    g = {"__name__": "node_under_test"}
    exec(compile(node, "<node>", "exec"), g)
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    return routes, g


try:
    routes, node = boot_node()
except Exception as e:
    routes, node = {}, {}
    check("节点模板能装载", False, "%s: %s" % (type(e).__name__, e))
if routes:
    check("节点模板能装载", True)
    for want in (("GET", "/qwen/console"), ("GET", "/qwen/console.json"), ("GET", "/qwen/steps/{tid}"),
                 ("GET", "/qwen/preview/{tid}"), ("POST", "/qwen/decrypt")):
        check("模板里注册了 %s %s" % want, want in routes)
    # 端上封的一封信，云端要能原地解开：这是「终端显示生成了什么」的地基
    import collections
    got = node["_unpack"](_PACK(collections))
    check("信封在模板里解得开", got.get("steps") == 20 and got.get("width") == 1024, got)
    st = node["_envelope_stats"](_PACK(collections))
    check("回显字段名对得上", [f["k"] for f in st["fields"]][:3] == ["prompt", "steps", "width"], st)
    check("回显不含字段值", "一只穿西装的猫" not in json.dumps(st, ensure_ascii=False))

print()
print("ALL OK" if not FAIL else "FAILED: %s" % ", ".join(FAIL))
sys.exit(0 if not FAIL else 1)
