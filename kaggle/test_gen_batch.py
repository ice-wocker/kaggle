# -*- coding: utf-8 -*-
"""gen_batch.py 的自检：不碰网络，验信封与云端同一实现。

加密走的是同一条链，但封包字段顺序一旦和 verify_seal.py 不一致，云端只会回
「bad envelope」，而那是发生在等了十几分钟之后的。所以这里当场对齐：
  python3 kaggle/test_gen_batch.py
"""
import collections, hashlib, json, os, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_batch as gb
import verify_seal as vs   # 独立实现，跑起来就是它的自检

DEV_HEX = "cb210100005892cb064e14e381e64030b57ac39c"   # 设备日志里那条本地封装
PROMPT = "一只穿西装的猫，赛博朋克城市夜景，电影感打光，85mm 浅景深"
FAIL = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else "  " + str(detail)))
    if not ok:
        FAIL.append(name)


k = gb.key()
# 1) 与设备日志逐字节一致
env1 = vs.encodeTask(PROMPT, k) if hasattr(vs, "encodeTask") else None
check("encodeTask 与设备日志同前缀", env1.hex()[:40] == DEV_HEX, env1.hex()[:40])

# 2) 信封：gen_batch.pack_task 与 verify_seal.pack_task 逐字节一致
f = collections.OrderedDict([
    ("prompt", PROMPT), ("negative", ""), ("steps", 20), ("cfg", 1.0),
    ("seed", 20261025), ("batch", 1), ("width", 1024), ("height", 1024), ("nsfw", False),
])
a = gb.pack_task(f, k)
b = vs.pack_task(f, k)
check("pack_task 与 verify_seal 逐字节一致", a == b, a.hex()[:32] + " vs " + b.hex()[:32])

# 3) 解封回来字段全等（用引擎自己的解封实现，不是我们自己写的）
src = open(os.path.join(HERE, "qwen_setup.py"), encoding="utf-8").read()
ns = {"hashlib": hashlib, "struct": struct}
exec(src[src.index("def _keystream"):src.index("def pack_task")], ns)
inner = ns["_unfeistel"](a[6:], k)
got, i = {}, 2
while i < len(inner):
    kl = inner[i]; i += 1
    name = inner[i:i + kl].decode(); i += kl
    vl = int.from_bytes(inner[i:i + 2], "big"); i += 2
    typ = inner[i]; i += 1
    raw = inner[i:i + vl]; i += vl
    got[name] = (raw == b"\x01") if typ == 0 else (
        int.from_bytes(raw, "big") * (-1 if typ == 2 else 1)) if typ in (1, 2) else (
        struct.unpack(">d", raw)[0] if typ == 3 else raw.decode())
check("云端 _unfeistel 解回字段全等", got == dict(f), got)
check("信封上限 65535 内", len(a) < 65536, len(a))

# 4) 9 张的计划：种子不重、信封都不超长
plan = [gb.envelope(gb.PROMPT, s, 20, 1024, 1024) for s in gb.SEEDS]
check("9 个种子互不相同", len(set(gb.SEEDS)) == len(gb.SEEDS))
check("9 个信封都在 300B 以内", all(len(e) < 300 for e in plan), max(map(len, plan)))

print()
print("ALL OK" if not FAIL else "FAILED: %s" % FAIL)
sys.exit(1 if FAIL else 0)
