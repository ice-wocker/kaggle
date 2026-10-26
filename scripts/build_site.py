#!/usr/bin/env python3
"""把 README.md 渲染成 GitHub Pages 静态站，输出到 site/。

site/assets/style.css 是手写源文件，其余都是生成物。

    pip install markdown && python3 scripts/build_site.py
"""
import re
import shutil
import pathlib
import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
TITLE = "Qwen-Image-2.1 Uncensored · 一键出图"
NAV = [("终端", "#_1"), ("APK", "#apk"), ("加密链路", "#_5")]

BENCH_PLACEHOLDER = "_9 张批量还没有真数字，仓库里先留白。跑完 `kaggle/gen_batch.py` 再回填。_"


def main() -> None:
    src = (ROOT / "README.md").read_text(encoding="utf-8")
    src = src.replace("<!-- BENCH:TABLE -->", BENCH_PLACEHOLDER)

    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"])
    body = md.convert(src)

    (SITE / "assets").mkdir(parents=True, exist_ok=True)
    for name in ("docs", "apk"):
        dst = SITE / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(ROOT / name, dst)

    nav = "".join(f'<a href="{href}">{label}</a>' for label, href in NAV)
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="Kaggle 免费 GPU 跑 Qwen-Image-2.1-Uncensored INT8，Android 一个按钮走完全流程。">
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<header class="top">
  <div class="wrap">
    <a class="brand" href="./">Qwen-Image-2.1 <span>Uncensored</span></a>
    <nav>{nav}</nav>
  </div>
</header>
<div class="layout wrap">
  <aside class="toc"><p>目录</p>{md.toc}</aside>
  <main class="prose">
{body}
  </main>
</div>
<footer><div class="wrap">静态页面，由仓库 README 生成 · GitHub Pages</div></footer>
</body>
</html>
"""
    if not (SITE / "assets" / "style.css").exists():
        raise SystemExit("缺 site/assets/style.css")
    (SITE / "index.html").write_text(page, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    (SITE / "404.html").write_text(
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>404</title>'
        '<link rel="stylesheet" href="/assets/style.css"></head><body><div class="wrap prose">'
        '<h1>404</h1><p>没有这个页面。<a href="/">回首页</a></p></div></body></html>',
        encoding="utf-8",
    )
    print(f"site/index.html {(SITE / 'index.html').stat().st_size} bytes")


if __name__ == "__main__":
    main()
