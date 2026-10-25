# Qwen-Image-2.1 Uncensored · 一键出图

Kaggle 免费 GPU 跑 `Qwen-Image-2.1-Uncensored` 的 **INT8** 量化版，Android 一个按钮走完全流程。

![实测出图](docs/sample-cat.png)

*T4 上 20 步 1024×1024，236 秒。上面的图就是 `docs/run_summary.json` 里那次运行的产物。*

## 目录

```
kaggle/          Kaggle 端：ComfyUI + 加密节点 + 出图/常驻服务
android/         APK 源码（纯 Java，不依赖 AndroidX/Gradle）+ 一条命令构建
apk/             已签名安装包（预编译，直接装）
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

一般不用手动跑，APK 会自己 push。想自己起或调试时：

```bash
cd kaggle && kaggle kernels push
```

加速器 **GPU T4**，Internet 打开。跑起来后依次做：装环境 → 拉权重 → 起 ComfyUI →
装加密节点 → 闭环自检（密文图 + 任务信封双向验证）→ 开公网隧道 → 把地址发布到日志。

日志里那行是给手机看的：

```
QWEN21://<base64url({"kind":"ready","url":"https://xxxx.trycloudflare.com","token":"QW21-…"})>
```

为什么走日志：Kaggle 没有「读某个 kernel 最新版本输出文件」的公开 API，但有日志。
地址是唯一必须由云端告诉手机的变量（Quick Tunnel 每次重启都换域名）。
想要固定域名，把 `NGROK_TOKEN` 放进 Kaggle Secrets，会自动优先走 ngrok。

两个容易被忽略的设置：

- **大件都放 /tmp**。Kaggle 保存版本时 output 有 ~20GB 上限，17GB 权重放在 working
  会让保存直接失败，日志也跟着丢，自续跑就断了。
- **依赖不重装**。镜像自带 torch/numpy/transformers/aiohttp/pillow/safetensors/einops，
  自定义节点只用得到 numpy/torch/PIL，所以 `pip` 只补缺的那几个，绝不碰 torch。

## APK（已编译，直接装）

安装包在仓库里：**[`apk/qwen-image-2.1-oneclick.apk`](apk/qwen-image-2.1-oneclick.apk)**（81 KB，minSdk 24，已签名）。

**装好只需要填一个 Kaggle API Key。** 没有云端地址，没有口令，没有 push 命令，没有构建脚本。

点一次「一键出图」，手机自己干这些事：

1. `hello` 探出 Key 属于哪个账号（不用你填用户名）
2. `kernels/push` 起或复用你名下的 GPU 会话（内核代码编译进 APK，离线也能推）
3. 轮询内核日志，等云端把隧道地址和设备令牌写下来（`QWEN21://…` 那一行）
4. 探活通过后，端上封一个加密信封提交出图
5. 取回图片，存相册

首次约 10-15 分钟（装 ComfyUI + 拉 17GB INT8 权重），之后每次开 App 秒连；
云端快到期时会自己 push 新版本续跑，手机不用管。

装好后还能调：风格预设 8 种、NSFW 开关、补词开关、实际提示词预览、
步数 / 宽高 / CFG / 种子（定值可复现）、一次 1/2/4 张、缩略图切换、存相册（`Pictures/Qwen21/`）。

### 自己编译

```bash
cd android && ./build.sh
```

只要机器上有 `javac`，SDK 缺失会自动下载到 `android/.sdk/`，产出直接覆盖 `apk/` 里那一份。
签名用固定的 `android/.keystore`（首次自动生成），所以升级安装不会因签名变化被系统拒掉。

## 加密链路

`comfy_node_crypt` 是 ComfyUI 原生 custom node，不是外挂：

- `EncryptText`：明文 → 4 轮 ARX Feistel（8 字节分组 PKCS#7）→ 每个字节展开成 16 位码字 → 位反转铺进 32×32 起的方形图
- `DecryptText`：从图里读回码字，逆向还原
- APK 的 `Crypto.java` 是独立实现，seed 编译进包里，明文不出设备

上一版只把提示词装进密文，步数、尺寸、种子、负面词还是明文 JSON 过网。现在整条指令走一个信封：

```
CB 22 01 00 | len(2) | Feistel( b"Q2" | n × [klen key vlen type payload] )
```

云端 `/qwen/generate` 现拆现拼工作流，解密后的提示词只活在进程内存里，
不打印、不落盘。`/qwen/health`、`/qwen/task/{id}` 只回状态与文件名。

边长取 2 的幂，位反转才是严格双射；码字多时边长自动翻倍。

![密文图](docs/sample-cipher.png)

*72 个码字铺在 32×32 里。上面这张就是这次实测的密文图，肉眼读不出内容。*

已做双向交叉验证：Java 码字 → Python 图 → Java 读回，以及反向；中英日韩 + emoji + 换行 + 符号全部字节级一致。

## 已知边界

- **首次要等**：装环境 + 17GB 权重约 10-15 分钟，这不是 APK 能优化的部分。
- **单次会话上限 12 小时**：到期前云端自己 push 新版本接棒，旧会话优雅退出。
  重启后要重新下权重（这就是上面说的 /tmp 取舍换来的可靠续跑）。
- **没在真机 + 真 GPU 上端到端跑过**：本次只验证到
  ①Java↔Python 信封字节级互验 ②APK 内嵌内核 vs 仓库源码 sha256 一致
  ③真实 aiohttp 服务上跑通「加密上传 → 解封 → 拼工作流 → 出图 → 取回」
  ④节点路由单测（鉴权 401 / 排队 / 完成）。推理那一步需要 T4。
