package com.cb.qwen21;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.method.ScrollingMovementMethod;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends Activity {

    /** 内置 seed（与 Kaggle 端一致），种子不落盘也不上网络 */
    private static final String SEED = "qwen21unc2026";

    /** 出图参数的默认值：与 docs/run_summary.json 那次实测完全一致，装上就能复现。 */
    static final String D_PROMPT = "一只穿西装的猫，赛博朋克城市夜景，电影感打光，85mm 浅景深";
    static final int D_STEPS = 20;
    static final int D_SIZE = 1024;
    static final double D_CFG = 1.0;
    static final long D_SEED = 20261025L;

    private final ExecutorService pool = Executors.newFixedThreadPool(3);
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final AtomicBoolean cancelled = new AtomicBoolean(false);
    private final List<Bitmap> results = new ArrayList<>();

    private SharedPreferences sp;
    private EditText etPrompt, etNeg, etSteps, etW, etH, etCfg, etSeed;
    private static final String[] KEY_LABELS = {"KAGGLE_KEY（账号=用户名）", "KGAT_（账号由接口探测）"};
    private EditText etKey, etUser;
    private Button btnKeyMode;
    private int keyMode = 0;
    private TextView tvStatus, tvLog, tvTaskHex, tvPromptPreview, tvCloud;
    private LinearLayout gallery, statusDot;
    private ImageView preview;
    private Button btnRun, btnStyle, btnSave, btnBatch;
    private int batch = 1;
    private android.widget.CompoundButton swNsfw, swQuality;
    private String styleName = "自动";
    private byte[] lastPng;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        sp = getSharedPreferences("qwen21", MODE_PRIVATE);
        // 从旧版升级上来的：地址/口令已无用，Key 还没填过就直接弹配置
        if (!sp.getBoolean("offline", false)) sp.edit().putBoolean("offline", true).apply();
        batch = Math.max(1, Math.min(4, sp.getInt("batch", 1)));
        setContentView(buildUi());
        renderTasks();
    }

    // ---------------- UI ----------------
    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(Ui.BG);
        LinearLayout root = Ui.column(this);
        int pad = Ui.dp(this, 16);
        root.setPadding(pad, pad, pad, Ui.dp(this, 32));
        scroll.addView(root);

        // Header
        LinearLayout head = Ui.column(this);
        TextView title = Ui.label(this, "Qwen-Image 2.1 · 一键出图", 22, Ui.TEXT);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        head.addView(title);
        LinearLayout dotRow = new LinearLayout(this);
        dotRow.setOrientation(LinearLayout.HORIZONTAL);
        dotRow.setGravity(Gravity.CENTER_VERTICAL);
        statusDot = new LinearLayout(this);
        View circle = new View(this);
        circle.setBackground(Ui.rounded(this, 0xFF6B7280, 99, 0));
        statusDot.addView(circle, new LinearLayout.LayoutParams(Ui.dp(this, 8), Ui.dp(this, 8)));
        dotRow.addView(statusDot);
        tvStatus = Ui.label(this, "未连接", 13, Ui.MUTED);
        LinearLayout.LayoutParams slp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        slp.leftMargin = Ui.dp(this, 8);
        dotRow.addView(tvStatus, slp);
        Ui.add(head, dotRow, 6);
        TextView sub = Ui.label(this, "装好填地址即用 · 云端 T4 全程免跑脚本", 12, Ui.MUTED);
        Ui.add(head, sub, 4);
        Ui.add(root, head, 0);

        // ① 只需一个 Key
        LinearLayout conn = Ui.panel(this);
        conn.addView(Ui.card(this, "① 一个 Key，剩下全自动"));
        TextView tip = Ui.label(this, "Key 只存在本机，仅发给 kaggle.com 与你自己启动的那台 GPU。"
                + "首次约 10-15 分钟（装环境 + 下 17GB INT8 权重），之后每次开 App 秒连。", 11, Ui.MUTED);
        Ui.add(conn, tip, 6);
        etUser = Ui.input(this, "Kaggle 用户名（新版 KGAT_ 令牌可留空，自动探测）",
                sp.getString("user", ""), false);
        Ui.add(conn, etUser, 10);
        etKey = Ui.input(this, "粘 KGAT_… 令牌（或旧版 KAGGLE_KEY）", sp.getString("key", ""), false);
        Ui.add(conn, etKey, 8);
        btnKeyMode = btn("Key 形式：" + KEY_LABELS[keyMode]);
        btnKeyMode.setOnClickListener(v -> pickKeyMode());
        Ui.add(conn, btnKeyMode, 8);
        tvCloud = Ui.label(this, "云端未启动", 12, Ui.MUTED);
        Ui.add(conn, tvCloud, 10);
        Ui.add(root, conn, 14);

        // 任务面板
        LinearLayout task = Ui.panel(this);
        task.addView(Ui.card(this, "② 输入任务"));
        etPrompt = Ui.input(this, "想要什么图？一句中文也行",
                sp.getString("prompt", D_PROMPT), true);
        Ui.add(task, etPrompt, 10);

        // 风格 + NSFW + 自动补词
        LinearLayout optRow = new LinearLayout(this);
        optRow.setOrientation(LinearLayout.HORIZONTAL);
        optRow.setGravity(Gravity.CENTER_VERTICAL);
        btnStyle = btn("风格：" + styleName);
        btnStyle.setOnClickListener(v -> pickStyle());
        optRow.addView(btnStyle, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1.4f));

        LinearLayout swWrap = Ui.column(this);
        swNsfw = new android.widget.Switch(this);
        swNsfw.setText("NSFW");
        swNsfw.setTextSize(13);
        swNsfw.setTextColor(Ui.TEXT);
        swNsfw.setChecked(sp.getBoolean("nsfw", true));
        swWrap.addView(swNsfw);
        swQuality = new android.widget.Switch(this);
        swQuality.setText("补词");
        swQuality.setTextSize(13);
        swQuality.setTextColor(Ui.TEXT);
        swQuality.setChecked(sp.getBoolean("quality", true));
        swWrap.addView(swQuality);
        optRow.addView(swWrap, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Ui.add(task, optRow, 10);

        tvPromptPreview = Ui.label(this, "", 11, Ui.MUTED);
        tvPromptPreview.setMaxLines(4);
        Ui.add(task, tvPromptPreview, 8);

        etNeg = Ui.input(this, "负面词（留空用内置干净版）", sp.getString("neg", ""), true);
        etNeg.setMinLines(1);
        Ui.add(task, etNeg, 8);

        LinearLayout nums = new LinearLayout(this);
        nums.setOrientation(LinearLayout.HORIZONTAL);
        etSteps = Ui.input(this, "步数", sp.getString("steps", String.valueOf(D_STEPS)), true);
        etW = Ui.input(this, "宽", sp.getString("w", String.valueOf(D_SIZE)), true);
        etH = Ui.input(this, "高", sp.getString("h", String.valueOf(D_SIZE)), true);
        LinearLayout.LayoutParams np = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        np.rightMargin = Ui.dp(this, 8);
        nums.addView(wrapField("步数", etSteps), np);
        nums.addView(wrapField("宽", etW), np);
        nums.addView(wrapField("高", etH), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Ui.add(task, nums, 10);

        LinearLayout nums2 = new LinearLayout(this);
        nums2.setOrientation(LinearLayout.HORIZONTAL);
        etCfg = Ui.input(this, "cfg", sp.getString("cfg", String.valueOf(D_CFG)), true);
        etSeed = Ui.input(this, "种子(-1 随机)", sp.getString("seed", String.valueOf(D_SEED)), true);
        LinearLayout.LayoutParams np2 = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        np2.rightMargin = Ui.dp(this, 8);
        nums2.addView(wrapField("CFG", etCfg), np2);
        nums2.addView(wrapField("种子", etSeed),
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Ui.add(task, nums2, 8);

        tvTaskHex = Ui.label(this, "", 11, Ui.MUTED);
        tvTaskHex.setTypeface(android.graphics.Typeface.MONOSPACE);
        tvTaskHex.setMaxLines(3);
        Ui.add(task, tvTaskHex, 10);
        etPrompt.setOnFocusChangeListener((v, f) -> { if (!f) renderTasks(); });
        etPrompt.addTextChangedListener(new android.text.TextWatcher() {
            public void beforeTextChanged(CharSequence s, int a, int b, int c) {}
            public void onTextChanged(CharSequence s, int a, int b, int c) {}
            public void afterTextChanged(android.text.Editable s) { renderTasks(); }
        });

        btnRun = new Button(this);
        btnRun.setText("一键出图");
        btnRun.setTextSize(17);
        btnRun.setTextColor(0xFF07100E);
        btnRun.setAllCaps(false);
        btnRun.setBackground(Ui.rounded(this, Ui.ACCENT, 14, 0));
        btnRun.setOnClickListener(v -> {
            if (cancelled.get()) return;
            generate();
        });
        Ui.add(task, btnRun, 14);
        Ui.add(root, task, 14);

        // 结果
        LinearLayout res = Ui.panel(this);
        res.addView(Ui.card(this, "③ 结果"));
        LinearLayout batchRow = new LinearLayout(this);
        batchRow.setOrientation(LinearLayout.HORIZONTAL);
        batchRow.setGravity(Gravity.CENTER_VERTICAL);
        btnBatch = btn("一次 " + batch + " 张");
        btnBatch.setOnClickListener(v -> pickBatch());
        batchRow.addView(btnBatch, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Ui.add(res, batchRow, 10);

        preview = new ImageView(this);
        preview.setAdjustViewBounds(true);
        preview.setScaleType(ImageView.ScaleType.FIT_CENTER);
        preview.setBackground(Ui.rounded(this, 0xFF0B0F14, 12, Ui.LINE));
        preview.setMinimumHeight(Ui.dp(this, 200));
        Ui.add(res, preview, 10);
        btnSave = btn("保存到相册");
        btnSave.setOnClickListener(v -> saveToGallery());
        Ui.add(res, btnSave, 8);
        gallery = Ui.column(this);
        Ui.add(res, gallery, 10);
        Ui.add(root, res, 14);

        // 日志
        LinearLayout logs = Ui.panel(this);
        logs.addView(Ui.card(this, "运行日志"));
        tvLog = Ui.label(this, "就绪", 12, Ui.MUTED);
        tvLog.setTypeface(android.graphics.Typeface.MONOSPACE);
        tvLog.setMovementMethod(new ScrollingMovementMethod());
        Ui.add(logs, tvLog, 8);
        Ui.add(root, logs, 14);

        return scroll;
    }

    private LinearLayout wrapField(String name, EditText e) {
        LinearLayout l = Ui.column(this);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        l.addView(Ui.label(this, name, 11, Ui.MUTED), lp);
        l.addView(e, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        return l;
    }

    private Button btn(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setTextColor(Ui.TEXT);
        b.setBackground(Ui.rounded(this, 0xFF1F2732, 12, Ui.LINE));
        return b;
    }

    private static LinearLayout.LayoutParams weight(float w) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, w);
        lp.rightMargin = 12;
        return lp;
    }

    // ---------------- 逻辑 ----------------
    private byte[] key() {
        return Crypto.key(SEED);
    }

    private void renderTasks() {
        try {
            String raw = etPrompt.getText().toString().trim();
            if (raw.isEmpty()) {
                tvTaskHex.setText("等待输入任务…");
                if (tvPromptPreview != null) tvPromptPreview.setText("");
                return;
            }
            String full = Presets.compose(raw, styleName, swQuality != null && swQuality.isChecked());
            if (tvPromptPreview != null) tvPromptPreview.setText("实际提示词：" + full);
            byte[] blob = Crypto.encodeTask(raw, key());
            tvTaskHex.setText("本地加密 " + blob.length + "B · " + Crypto.hex(blob).substring(0, 40) + "…");
        } catch (Exception e) {
            tvTaskHex.setText("加密失败: " + e.getMessage());
        }
    }

    private void pickStyle() {
        new AlertDialog.Builder(this)
                .setTitle("风格取向")
                .setItems(Presets.STYLE_NAMES, (d, i) -> {
                    styleName = Presets.STYLE_NAMES[i];
                    btnStyle.setText("风格：" + styleName);
                    renderTasks();
                })
                .show();
    }

    private void pickBatch() {
        final String[] opts = {"1 张", "2 张", "4 张"};
        final int[] vals = {1, 2, 4};
        int cur = 0;
        for (int i = 0; i < vals.length; i++) if (vals[i] == batch) cur = i;
        new AlertDialog.Builder(this)
                .setTitle("每批张数（越多越慢）")
                .setSingleChoiceItems(opts, cur, (d, i) -> {
                    batch = vals[i];
                    btnBatch.setText("一次 " + batch + " 张");
                    sp.edit().putInt("batch", batch).apply();
                    d.dismiss();
                })
                .show();
    }

    /** 存到相册（MediaStore，无需存储权限）。 */
    private void saveToGallery() {
        if (lastPng == null) {
            log("还没有可保存的图");
            return;
        }
        pool.execute(() -> {
            try {
                String name = "qwen21_" + System.currentTimeMillis() + ".png";
                android.content.ContentValues cv = new android.content.ContentValues();
                cv.put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, name);
                cv.put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/png");
                if (android.os.Build.VERSION.SDK_INT >= 29)
                    cv.put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Qwen21");
                Uri uri = getContentResolver().insert(
                        android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv);
                if (uri == null) throw new IllegalStateException("相册不可写");
                java.io.OutputStream os = getContentResolver().openOutputStream(uri);
                os.write(lastPng);
                os.close();
                log("已保存到相册: Pictures/Qwen21/" + name);
            } catch (Exception e) {
                log("保存失败: " + e.getMessage());
            }
        });
    }

    private void log(String s) {
        ui.post(() -> tvLog.setText(s + "\n" + tvLog.getText()));
    }

    private void setStatus(String s, int color) {
        ui.post(() -> {
            tvStatus.setText(s);
            View c = statusDot.getChildAt(0);
            c.setBackground(Ui.rounded(this, color, 99, 0));
        });
    }

    private Api api() {
        // 云端地址由 Kaggle.java 自己拿到并缓存在 sp 里；这里只负责取用
        return apiCached();
    }

    private Api apiCached() {
        String u = sp.getString("url", "");
        if (u.isEmpty()) throw new IllegalStateException("云端还没起，点「一键出图」会自动启动");
        return new Api(u, sp.getString("token", ""));
    }

    private void pickKeyMode() {
        new AlertDialog.Builder(this)
                .setTitle("Key 形式")
                .setSingleChoiceItems(KEY_LABELS, keyMode, (d, i) -> {
                    keyMode = i;
                    btnKeyMode.setText("Key 形式：" + KEY_LABELS[i]);
                    etUser.setHint(i == 1 ? "可留空，自动探测" : "Kaggle 用户名");
                    d.dismiss();
                })
                .show();
    }

    private void saveKeyFields() {
        String k = Kaggle.normalizeKey(etKey.getText().toString());
        // 顺手把粘进来的 export 行、网页噪声洗掉，回填给人看得见的干净值
        if (!k.isEmpty() && !k.equals(etKey.getText().toString().trim())) {
            etKey.setText(k);
            etKey.setSelection(k.length());
        }
        sp.edit().putString("key", k)
                .putString("user", etUser.getText().toString().trim()).apply();
    }

    /**
     * 全链路：Key -> 起/复用 GPU 会话 -> 等地址下发 -> 探活。
     * 云端已经在线时这一步几乎瞬时，所以每次出图前都跑一遍也没成本。
     */
    private void ensureCloud() throws Exception {
        String base = sp.getString("url", "");
        String token = sp.getString("token", "");
        if (!base.isEmpty()) {
            try {
                JSONObject h = Kaggle.probe(base, token);
                if (h.optBoolean("ok") && h.optBoolean("auth", true)) {
                    ui.post(() -> {
                        tvCloud.setText("云端在线 · " + base.replace("https://", ""));
                        setStatus("在线", Ui.ACCENT);
                    });
                    return;
                }
            } catch (Exception e) {
                log("旧地址失效，重新拉起云端（能连上就不用管 Key）");
            }
        }
        final String key = Kaggle.normalizeKey(etKey.getText().toString());
        final String usr = etUser.getText().toString().trim();
        if (key.isEmpty()) throw new IllegalStateException("先填 Kaggle API Key（KGAT_ 开头那个令牌，粘 export 整行也行）");
        if (!Kaggle.complete(usr, key))
            throw new IllegalStateException("这是旧式 KAGGLE_KEY，还得补上 Kaggle 用户名；新版 KGAT_ 令牌不用填");
        final String effUser = usr.isEmpty() ? "me" : usr;
        ui.post(() -> setStatus("拉起云端…", 0xFFE0B341));
        Kaggle.Progress p = new Kaggle.Progress() {
            @Override public void stage(String st) {
                ui.post(() -> {
                    tvCloud.setText(st);
                    tvStatus.setText(st);
                });
                log("云 · " + st);
            }
            @Override public void detail(String d) {
                ui.post(() -> tvCloud.setText(d));
            }
            @Override public boolean cancelled() { return cancelled.get(); }
        };
        java.util.Map<String, String> got = new Kaggle(effUser, key).bringUp(key, p, 20 * 60 * 1000L);
        final String url = got.get("url"), tok = got.get("token");
        sp.edit().putString("url", url).putString("token", tok)
                .putString("owner", got.get("owner")).apply();
        ui.post(() -> {
            tvCloud.setText("云端在线 · " + url.replace("https://", ""));
            setStatus("在线", Ui.ACCENT);
        });
        log("云 · 地址已下发 " + url);
    }

    private void generate() {
        cancelled.set(false);
        btnRun.setEnabled(false);
        btnRun.setText("准备中…");
        final String rawPrompt = etPrompt.getText().toString().trim();
        if (rawPrompt.isEmpty()) {
            toastLog("任务为空");
            resetBtn();
            return;
        }
        saveKeyFields();
        sp.edit().putString("prompt", rawPrompt)
                .putString("steps", etSteps.getText().toString())
                .putString("w", etW.getText().toString())
                .putString("h", etH.getText().toString())
                .putString("cfg", etCfg.getText().toString())
                .putString("seed", etSeed.getText().toString())
                .putString("neg", etNeg.getText().toString())
                .putBoolean("nsfw", swNsfw.isChecked())
                .putBoolean("quality", swQuality.isChecked()).apply();

        final int steps = parseInt(etSteps.getText().toString(), 20);
        final int w = align(parseInt(etW.getText().toString(), 1024));
        final int h = align(parseInt(etH.getText().toString(), 1024));
        final double cfg = parseD(etCfg.getText().toString(), 1.0);
        final long seed = parseLong(etSeed.getText().toString(), -1L);
        final boolean nsfw = swNsfw.isChecked();

        final String prompt = Presets.compose(rawPrompt, styleName, swQuality.isChecked());
        String negRaw = etNeg.getText().toString().trim();
        if (negRaw.isEmpty()) negRaw = nsfw ? Presets.nsfwNegative() : "";
        final String neg = nsfw ? Presets.stripCensor(negRaw) : negRaw;
        final long useSeed = seed >= 0 ? seed : Math.abs(new java.util.Random().nextLong() % 1_000_000_000L);

        setStatus("准备中…", 0xFFE0B341);
        log("— 新任务 —");

        pool.execute(() -> {
            try {
                // 1) 云端：Key 换地址，已经在线就秒过
                ensureCloud();
                Api a = api();

                // 2) 端上封一个信封：提示词 + 全部参数一起加密，明文不出设备
                java.util.LinkedHashMap<String, Object> f = new java.util.LinkedHashMap<>();
                f.put("prompt", prompt);
                f.put("negative", neg);
                f.put("steps", steps);
                f.put("cfg", cfg);
                f.put("seed", useSeed);
                f.put("batch", batch);
                f.put("width", w);
                f.put("height", h);
                f.put("nsfw", nsfw);
                byte[] env = Crypto.packTask(f, key());
                log("① 本地封装 " + env.length + "B · " + Crypto.hex(env).substring(0, 32) + "…");
                log("参数 " + w + "x" + h + " · " + steps + " 步 · cfg " + cfg + " · seed " + useSeed
                        + (nsfw ? " · NSFW 开" : ""));

                // 3) 加密提交
                String tid = a.submitSealed(env);
                log("② 已入队 " + tid.substring(0, 8));
                afterSubmit(tid, a, env);

            } catch (Api.AuthError e) {
                setStatus("鉴权没过", 0xFFE05B5B);
                // Kaggle 那句 "Key 无效" 太笼统：设备令牌、账号归属、云端令牌都可能
                log("出错: " + (e.getMessage() == null ? "鉴权没过" : e.getMessage()));
                if (e.getMessage() != null && e.getMessage().contains("Kaggle"))
                    log("  提示: 新版页面只给一个 KGAT_ 令牌，账号由它自己认，用户名可以留空");
                resetBtn();
            } catch (Exception e) {
                setStatus("失败", 0xFFE05B5B);
                log("出错: " + e.getMessage());
                resetBtn();
            }
        });
    }

    private void afterSubmit(String tid, Api a, byte[] env) throws Exception {
        ui.post(() -> {
            setStatus("生成中…", 0xFFE0B341);
            btnRun.setText("生成中…");
        });
        log("③ 等待 GPU（T4 上 1024×1024/20 步约 4 分钟）");
        long t0 = System.currentTimeMillis();
        JSONObject st = null;
        while (true) {
            if (cancelled.get()) throw new InterruptedException("已取消");
            st = a.task(tid);
            String state = st.optString("state");
            if ("done".equals(state)) break;
            if ("error".equals(state)) throw new IllegalStateException("云端报错: " + st.optString("detail"));
            long sec = (System.currentTimeMillis() - t0) / 1000;
            final long fs = sec;
            ui.post(() -> {
                setStatus("生成中 " + fs + "s", 0xFFE0B341);
                btnRun.setText("生成中 " + fs + "s");
            });
            Thread.sleep(2000);
        }

        JSONArray imgs = st.optJSONArray("images");
        if (imgs == null || imgs.length() == 0) throw new IllegalStateException("没有返回图片");
        java.util.List<Bitmap> got = new ArrayList<>();
        byte[] firstPng = null;
        for (int i = 0; i < imgs.length(); i++) {
            JSONObject im = imgs.getJSONObject(i);
            if (!"output".equals(im.optString("type", "output"))) continue;
            byte[] raw = a.view(im.optString("filename"), im.optString("subfolder"), im.optString("type", "output"));
            Bitmap bmp = BitmapFactory.decodeByteArray(raw, 0, raw.length);
            if (bmp == null) continue;
            if (firstPng == null) firstPng = raw;
            got.add(bmp);
        }
        if (got.isEmpty()) throw new IllegalStateException("图片解码失败");
        final byte[] png = firstPng;
        log("④ 出图 " + got.size() + " 张 " + got.get(0).getWidth() + "x" + got.get(0).getHeight()
                + " · 用时 " + st.optDouble("elapsed", 0) + "s · 信封 " + env.length + "B（全程未明文过网）");
        ui.post(() -> {
            for (int i = got.size() - 1; i >= 0; i--) results.add(0, got.get(i));
            lastPng = png;
            preview.setImageBitmap(got.get(0));
            rebuildGallery();
            setStatus("完成", Ui.ACCENT);
            resetBtn();
        });
    }

    private void rebuildGallery() {
        gallery.removeAllViews();
        for (int i = 0; i < Math.min(results.size(), 12); i++) {
            Bitmap bm = results.get(i);
            ImageView iv = new ImageView(this);
            int w = Ui.dp(this, 72), hh = Ui.dp(this, 72);
            iv.setLayoutParams(new LinearLayout.LayoutParams(w, hh));
            iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
            iv.setImageBitmap(bm);
            iv.setPadding(Ui.dp(this, 2), Ui.dp(this, 2), Ui.dp(this, 2), Ui.dp(this, 2));
            final Bitmap cur = bm;
            iv.setOnClickListener(v -> preview.setImageBitmap(cur));
            gallery.addView(iv);
        }
    }

    private void resetBtn() {
        ui.post(() -> {
            btnRun.setEnabled(true);
            btnRun.setText("一键出图");
        });
    }

    private void toastLog(String s) {
        log(s);
    }

    private static int parseInt(String s, int def) {
        try {
            return Integer.parseInt(s.trim());
        } catch (Exception e) {
            return def;
        }
    }

    private static double parseD(String s, double def) {
        try {
            double v = Double.parseDouble(s.trim());
            return v < 0 ? def : Math.min(30, v);
        } catch (Exception e) {
            return def;
        }
    }

    private static long parseLong(String s, long def) {
        try {
            long v = Long.parseLong(s.trim());
            return v < 0 ? def : v;
        } catch (Exception e) {
            return def;
        }
    }

    private static int align(int v) {
        int c = Math.max(256, Math.min(2048, v));
        return (c / 64) * 64;
    }

    @Override
    protected void onResume() {
        super.onResume();
        // 后台静默预热：已经有可用地址就不打扰用户，直接用
        if (!Kaggle.normalizeKey(etKey.getText().toString()).isEmpty()) {
            pool.execute(() -> {
                String base = sp.getString("url", "");
                if (base.isEmpty()) return;
                if (Kaggle.alive(base, sp.getString("token", "")))
                    ui.post(() -> {
                        tvCloud.setText("云端在线 · " + base.replace("https://", ""));
                        setStatus("在线", Ui.ACCENT);
                    });
            });
        }
    }

    @Override
    protected void onDestroy() {
        cancelled.set(true);
        pool.shutdownNow();
        super.onDestroy();
    }
}
