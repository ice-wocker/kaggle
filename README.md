# Qwen-Image-2.1 Uncensored · 一键出图

Kaggle 免费 GPU 跑 `Qwen-Image-2.1-Uncensored` 的 **INT8** 量化版，Android 一个按钮走完全流程。

![实测出图](docs/sample-cat.png)

*T4 上 20 步 1024×1024，236 秒。上面的图就是 `docs/run_summary.json` 里那次运行的产物。*

## 目录

```
kaggle/          Kaggle 端：ComfyUI + 加密节点 + 出图/常驻服务
android/         APK 源码（纯 Java，不依赖 AndroidX/Gradle）
apk/             已签名安装包
docs/            实测产物（出图、密文图、运行摘要）
```

## 模型

[abenzerps/Qwen-Image-2.1-Uncensored-Quantized](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized) 的 int8_convrot 三件套：

| 组件 | 文件 | 大小 |
| --- | --- | ---: |
| Diffusion | `qwen-image-2.1-UC-int8_convrot.safetensors` | 6.76 GB |
| Text Encoder | `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | 9.35 GB |
| VAE | `vae/qwen_image_2.1_vae_bf16.safetensors` | 676 MB |

T4 16G 够用：diffusion 进显存，TE 走 RAM——编码只发生一次，几乎不掉速。
走 hf-mirror 下载，17GB 约 2 分钟。

## Kaggle 端

```bash
export KAGGLE_API_TOKEN=<你的 KGAT>
cd kaggle && kaggle kernels push
```

加速器选 **GPU T4**，Internet 打开。跑完自动做三件事：

1. 加密节点闭环：`EncryptText` → 落盘密文图 → 从像素读回明文
2. 上传/读取链路探针
3. 真实出图（20 步，1024×1024）

`main_cell.py` 里 `Mode = "serve"` 可切成常驻服务 + trycloudflare 隧道，浏览器也能用。

## APK

```bash
cd android && ./build.sh    # 需要 javac + Android SDK 34 build-tools
```

装好后填云端地址、输任务、点「一键生成」。

## 加密链路

`comfy_node_crypt` 是 ComfyUI 原生 custom node，不是外挂：

- `EncryptText`：明文 → 4 轮 ARX Feistel（8 字节分组 PKCS#7）→ 每个字节展开成 16 位码字 → 位反转铺进 32×32 起的方形图
- `DecryptText`：从图里读回码字，逆向还原
- APK 的 `Crypto.java` 是独立实现，seed 编译进包里，明文不出设备

边长取 2 的幂，位反转才是严格双射；码字多时边长自动翻倍。

![密文图](docs/sample-cipher.png)

*72 个码字铺在 32×32 里。上面这张就是这次实测的密文图，肉眼读不出内容。*

已做双向交叉验证：Java 码字 → Python 图 → Java 读回，以及反向；中英日韩 + emoji + 换行 + 符号全部字节级一致。
