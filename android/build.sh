#!/usr/bin/env bash
# 一条命令出签名 APK：./build.sh
# 依赖：javac (JDK 11+)、Android SDK build-tools 34 + platform android-34。
# SDK 找不到时会自动下载到 .sdk/（约 200MB，仅首次）。
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
SRC="$ROOT/app"
OUT="$ROOT/../apk/qwen-image-2.1-oneclick.apk"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$ROOT/.sdk}}"
BT="$SDK/build-tools/34.0.0"
PLAT="$SDK/platforms/android-34/android.jar"
KS="$ROOT/.keystore"
KC="$ROOT/build/cert.pem"

need() { command -v "$1" >/dev/null 2>&1 || { echo "缺少 $1，请先安装" >&2; exit 1; }; }
need javac; need java
if [ ! -x "$BT/aapt2" ] || [ ! -f "$PLAT" ]; then
  echo ">>> 未找到 Android SDK，下载到 $SDK"
  mkdir -p "$SDK/build-tools" "$SDK/platforms"
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/bt.zip" https://dl.google.com/android/repository/build-tools_r34-linux.zip
  curl -fsSL -o "$tmp/plat.zip" https://dl.google.com/android/repository/platform-34-ext7_r03.zip
  python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$tmp/bt.zip" "$SDK/build-tools"
  mv "$SDK/build-tools/android-14" "$BT"
  python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$tmp/plat.zip" "$SDK/platforms"
  rm -rf "$tmp"
  chmod -R +x "$BT"
fi

# 固定签名，保证覆盖安装不会被拒。私钥不进仓库，所以按需二选一：
#   build/key.pem（私钥）+ build/cert.pem（证书）都在 -> 用这对配对；
#   否则自签一对。
# 这样换台机器重编，签名也不会变，用户不需要卸载再装。
if [ ! -f "$KS" ]; then
  if [ -f "$ROOT/build/key.pem" ] && [ -f "$KC" ]; then
    openssl pkcs12 -export -inkey "$ROOT/build/key.pem" -in "$KC" -name androiddebugkey \
      -passout pass:android -out "$KS" 2>/dev/null
  else
    keytool -genkeypair -keystore "$KS" -storepass android -keypass android \
      -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 \
      -dname "CN=Qwen21, O=CB, C=CN" >/dev/null 2>&1
  fi
fi

WORK="$ROOT/.build"
rm -rf "$WORK"; mkdir -p "$WORK/gen" "$WORK/obj" "$WORK/dex"

echo ">>> aapt2"
"$BT/aapt2" compile --dir "$SRC/res" -o "$WORK/res.zip"
"$BT/aapt2" link -o "$WORK/app-unsigned.apk" -I "$PLAT" \
  --manifest "$SRC/AndroidManifest.xml" -R "$WORK/res.zip" --java "$WORK/gen" \
  --min-sdk-version 24 --target-sdk-version 34 --auto-add-overlay

echo ">>> javac"
javac -nowarn -encoding UTF-8 --release 11 -classpath "$PLAT" -d "$WORK/obj" \
  $(find "$SRC/src" "$WORK/gen" -name '*.java')

echo ">>> d8"
"$BT/d8" --release --lib "$PLAT" --min-api 24 --output "$WORK/dex" \
  $(find "$WORK/obj" -name '*.class')

echo ">>> package & sign"
cd "$WORK"
cp app-unsigned.apk app.apk
python3 - "$WORK" <<'PY'
import sys, zipfile, os
w = sys.argv[1]
with zipfile.ZipFile(os.path.join(w, "app.apk"), "a", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(w, "dex", "classes.dex"), "classes.dex")
PY
"$BT/zipalign" -f -p 4 app.apk app-aligned.apk
mkdir -p "$(dirname "$OUT")"
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android \
  --ks-key-alias androiddebugkey --v1-signing-enabled true --min-sdk-version 24 \
  --out "$OUT" app-aligned.apk
"$BT/apksigner" verify --print-certs "$OUT" | head -2
ls -lh "$OUT"
echo "OK -> $OUT"
