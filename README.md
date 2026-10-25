# Qwen-Image-2.1 Uncensored · 一键出图

Kaggle 免费 GPU 跑 `Qwen-Image-2.1-Uncensored` 的 **INT8** 量化版，Android 一个按钮完成全部配置。

## 目录

```
kaggle/          Kaggle 端：ComfyUI + 加密节点 + 出图/常驻服务
android/         APK 源码（纯 Java，无 AndroidX 依赖）
apk/             已签名的安装包
```

## 模型

来源 [abenzerps/Qwen-Image-2.1-Uncensored-Quantized](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized)，int8_convrot 系列：

| 组件 | 文件 | 大小 |
| --- | --- | ---: |
| Diffusion | `qwen-image-2.1-UC-int8_convrot.safetensors` | 6.76 GB |
| Text Encoder | `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | 9.35 GB |
| VAE | `vae/qwen_image_2.1_vae_bf16.safetensors` | 676 MB |

T4 16G 够用：diffusion 进显存，TE 走 RAM（只跑一次编码，几乎不掉速）。

## Kaggle 端

推 notebook：

```bash
export KAGGLE_API_TOKEN=<你的 KGAT>
cd kaggle && kaggle kernels push
```

加速器选 **GPU T4**，Internet 打开。首跑约 15-20 分钟（16GB 权重），之后复用 output 秒起。

跑完会自检三件事：加密节点闭环、上传/读取链路、真实出图。

## APK

```bash
cd android && ./build.sh     # 需要 javac + Android SDK 34 build-tools
```

装好后：填云端地址 → 填任务 → 点「一键生成」。

## 加密链路

`comfy_node_crypt` 是 ComfyUI 原生 custom node，不是外挂脚本：

- `EncryptText`：文本 → 4 轮 ARX Feistel（8 字节分组 PKCS#7）→ 按 seed 派生的位置置换写进 16 位码字 → 位图
- `DecryptText`：逆向还原
- APK 侧 `Crypto.java` 是同算法的独立实现，seed 编译进包里，明文不出设备

Java 与 Python 两端已做交叉验证：中英日韩 + emoji + 换行 + 符号，双向字节级一致。

## 落地页

浏览器也能用：Kaggle 跑 `Mode="serve"` 拿 trycloudflare 地址，直接在上面输任务。
