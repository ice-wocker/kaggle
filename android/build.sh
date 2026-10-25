set -e
SDK=/opt/android-sdk
BT=$SDK/build-tools/34.0.0
PLAT=$SDK/platforms/android-34/android.jar
cd /tmp/apk/qwen21
rm -rf build; mkdir -p build/gen build/obj build/dex
$BT/aapt2 compile --dir res -o build/res.zip
$BT/aapt2 link -o build/app-unsigned.apk -I $PLAT --manifest AndroidManifest.xml \
   -R build/res.zip --java build/gen --min-sdk-version 24 --target-sdk-version 34 --auto-add-overlay
echo "--- javac ---"
javac -nowarn -encoding UTF-8 --release 11 -classpath $PLAT -d build/obj \
   $(find src build/gen -name '*.java') 2>&1 | cat
echo "--- d8 ---"
$BT/d8 --release --lib $PLAT --min-api 24 --output build/dex $(find build/obj -name '*.class')
echo "--- package ---"
cd build
cp app-unsigned.apk app.apk
cp res.zip res-final.zip
zip -q -j app.apk dex/classes.dex
$BT/zipalign -f -p 4 app.apk app-aligned.apk
$BT/apksigner sign --ks /tmp/apk/debug.keystore --ks-pass pass:android --key-pass pass:android \
   --ks-key-alias androiddebugkey --out /tmp/apk/qwen21.apk app-aligned.apk
$BT/apksigner verify --print-certs /tmp/apk/qwen21.apk | head -5
ls -lh /tmp/apk/qwen21.apk
