# -*- coding: utf-8 -*-
"""把两份源码拼成 notebook.ipynb，再把同一份 ipynb 烙进 Android 的 Kernel.java。

只认两个源头：qwen_setup.py（云端全部逻辑）+ run_notebook.py（启动序列）。
以前 ipynb 和 Kernel.java 各存一份拷贝，改了一边另一边忘掉，
手机上推上去的永远是旧云端代码——这才是「改了脚本还是老样子」的根。
用法：python3 gen_kernel.py [--check]
"""
import base64, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SETUP = os.path.join(HERE, "qwen_setup.py")
RUN = os.path.join(HERE, "run_notebook.py")
NBPATH = os.path.join(HERE, "notebook.ipynb")
JAVA = os.path.join(ROOT, "android", "app", "src", "com", "cb", "qwen21", "Kernel.java")
CHUNK = 60  # javac 的常量长度上限按 UTF-8 字节算，含中文的字面量留足余量


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def notebook_text(setup_src, run_src):
    cells = []
    for src in (setup_src, run_src):
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.splitlines(keepends=True),
        })
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    # 不写 indent：这份文件会被 base64 塞进 APK，体积要省
    return json.dumps(nb, ensure_ascii=False) + "\n"


def java_text(nb_json):
    b64 = base64.b64encode(nb_json.encode("utf-8")).decode()
    chunks = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)]
    body = ",\n".join('                "%s"' % c for c in chunks)
    return '''package com.cb.qwen21;

/**
 * 内置内核代码（kaggle/notebook.ipynb）。由 kaggle/gen_kernel.py 生成，**不要手改**。
 *
 * qwen_setup.py + run_notebook.py -> notebook.ipynb -> 这里。单一来源，
 * 改云端逻辑只需要改那两个 .py，然后跑一次 gen_kernel.py。
 *
 * 为什么切成 60 字符一段：javac 的「字符串常量过长」判断用的是编译期 UTF-8 字节数，
 * 不是字符数；含中文的常量叠到 3KB 就可能越线，而且它会连带把整个字面量算成一条。
 * 所以用数组运行时拼接，谁也别指望常量折叠。
 */
final class Kernel {

    private Kernel() {}

    private static final String B64 = joins();

    private static String joins() {
        String[] p = {
%s
        };
        StringBuilder sb = new StringBuilder(%d);
        for (String s : p) sb.append(s);
        return sb.toString();
    }

    static String notebook() {
        return new String(Auth.unb64(B64), java.nio.charset.StandardCharsets.UTF_8);
    }
}
''' % (body, len(b64))


def main():
    setup_src, run_src = read(SETUP), read(RUN)
    nb = notebook_text(setup_src, run_src)
    jv = java_text(nb)
    if "--check" in sys.argv:
        ok = True
        for path, want in ((NBPATH, nb), (JAVA, jv)):
            have = read(path)
            if have != want:
                ok = False
                print("过期: %s（跑一次 gen_kernel.py）" % os.path.relpath(path, ROOT))
        print("OK" if ok else "FAIL")
        return 0 if ok else 1
    with open(NBPATH, "w", encoding="utf-8") as f:
        f.write(nb)
    with open(JAVA, "w", encoding="utf-8") as f:
        f.write(jv)
    print("notebook.ipynb %d 字节 -> Kernel.java %d 字节" % (len(nb), len(jv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
