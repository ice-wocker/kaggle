# -*- coding: utf-8 -*-
"""把 docs/bench/batch.json 变成 README 里那张表。

数字只有 batch.json 一个来源：手抄进 README 的数字迟早跟实测脱节，
而这张表的全部意义就是「这些秒数是真跑出来的」。
  python3 kaggle/gen_bench_table.py            # 打印表格
  python3 kaggle/gen_bench_table.py --write    # 顺手替换 README 里的 <!-- BENCH:TABLE -->
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BATCH = os.path.join(ROOT, "docs", "bench", "batch.json")
README = os.path.join(ROOT, "README.md")
MARK = "<!-- BENCH:TABLE -->"


def table(d):
    rows = sorted(d["rows"], key=lambda r: r["n"])
    ok = [r for r in rows if r.get("ok")]
    L = ["| # | 种子 | 单张 | 尺寸 | 提示词 | 图 |", "| ---: | ---: | ---: | --- | --- | --- |"]
    for r in rows:
        if r.get("ok"):
            cell = "%.1fs" % r["elapsed_sec"]
            img = "![](bench/%s)" % r["file"]
            kb = "%d KB" % (r["bytes"] // 1024)
        else:
            cell, img, kb = "失败", "—", "—"
        L.append("| %d | %d | %s | %s | %s | %s |"
                 % (r["n"], r["seed"], cell, kb, r["prompt"][:18], img))
    L.append("")
    L.append("%d/%d 张，总墙钟 **%.0fs**（%.1f 分钟），单张 %.1fs，最快 %.1fs、最慢 %.1fs，"
             "共 %d 步 / %d×%d / cfg %s。"
             % (len(ok), d["count"], d["total_sec"], d["total_sec"] / 60,
                d.get("avg_sec") or 0, d.get("min_sec") or 0, d.get("max_sec") or 0,
                d["steps"], d["size"][0], d["size"][1], d["cfg"]))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(BATCH):
        print("还没有 %s：先跑一遍 kaggle/gen_batch.py。" % os.path.relpath(BATCH, ROOT), file=sys.stderr)
        return 2
    d = json.load(open(BATCH, encoding="utf-8"))
    t = table(d)
    if not a.write:
        print(t)
        return 0
    s = open(README, encoding="utf-8").read()
    if MARK not in s:
        print("README 里没有 %s，不知道该替换哪一段。" % MARK, file=sys.stderr)
        return 2
    s = s.replace(MARK, t, 1)
    open(README, "w", encoding="utf-8").write(s)
    print("已写入 README")
    return 0


if __name__ == "__main__":
    sys.exit(main())
