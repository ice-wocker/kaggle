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
    private static final String DEF_UNET = "qwen-image-2.1-UC-int8_convrot.safetensors";
    private static final String DEF_CLIP = "qwen3vl_8b_int8_convrot.safetensors";
    private static final String DEF_VAE = "qwen_image_2.1_vae_bf16.safetensors";

    private final ExecutorService pool = Executors.newFixedThreadPool(3);
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final AtomicBoolean cancelled = new AtomicBoolean(false);
    private final List<Bitmap> results = new ArrayList<>();

    private SharedPreferences sp;
    private EditText etUrl, etToken, etPrompt, etNeg, etSteps, etW, etH;
    private TextView tvStatus, tvLog, tvTaskHex;
    private LinearLayout gallery, statusDot;
    private ImageView preview;
    private Button btnRun;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        sp = getSharedPreferences("qwen21", MODE_PRIVATE);
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
        Ui.add(root, head, 0);

        // 连接面板
        LinearLayout conn = Ui.panel(this);
        conn.addView(Ui.card(this, "① 云端连接"));
        etUrl = Ui.input(this, "https://xxxx.trycloudflare.com", sp.getString("url", ""), false);
        Ui.add(conn, etUrl, 10);
        etToken = Ui.input(this, "访问令牌（可选，与云端约定的口令）", sp.getString("token", ""), false);
        Ui.add(conn, etToken, 8);
        LinearLayout connBtns = new LinearLayout(this);
        connBtns.setOrientation(LinearLayout.HORIZONTAL);
        Button bTest = btn("测试连接");
        bTest.setOnClickListener(v -> testConn());
        Button bSave = btn("保存并连接");
        bSave.setOnClickListener(v -> {
            sp.edit().putString("url", etUrl.getText().toString().trim())
                    .putString("token", etToken.getText().toString().trim()).apply();
            testConn();
        });
        connBtns.addView(bTest, weight(1));
        connBtns.addView(bSave, weight(1));
        Ui.add(conn, connBtns, 10);
        Ui.add(root, conn, 14);

        // 任务面板
        LinearLayout task = Ui.panel(this);
        task.addView(Ui.card(this, "② 输入任务"));
        etPrompt = Ui.input(this, "想要什么图？一句中文也行", sp.getString("prompt", ""), true);
        Ui.add(task, etPrompt, 10);
        etNeg = Ui.input(this, "反向提示词（一般留空）", "", true);
        etNeg.setMinLines(1);
        Ui.add(task, etNeg, 8);

        LinearLayout nums = new LinearLayout(this);
        nums.setOrientation(LinearLayout.HORIZONTAL);
        etSteps = Ui.input(this, "步数", sp.getString("steps", "20"), true);
        etW = Ui.input(this, "宽", sp.getString("w", "1024"), true);
        etH = Ui.input(this, "高", sp.getString("h", "1024"), true);
        LinearLayout.LayoutParams np = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        np.rightMargin = Ui.dp(this, 8);
        nums.addView(wrapField("步数", etSteps), np);
        nums.addView(wrapField("宽", etW), np);
        nums.addView(wrapField("高", etH), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Ui.add(task, nums, 10);

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
        btnRun.setText("一键生成");
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
        preview = new ImageView(this);
        preview.setAdjustViewBounds(true);
        preview.setScaleType(ImageView.ScaleType.FIT_CENTER);
        preview.setBackground(Ui.rounded(this, 0xFF0B0F14, 12, Ui.LINE));
        preview.setMinimumHeight(Ui.dp(this, 200));
        Ui.add(res, preview, 10);
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
            String p = etPrompt.getText().toString();
            if (p.isEmpty()) {
                tvTaskHex.setText("等待输入任务…");
                return;
            }
            byte[] blob = Crypto.encodeTask(p, key());
            tvTaskHex.setText("本地加密 " + blob.length + "B · " + Crypto.hex(blob).substring(0, 48) + "…");
        } catch (Exception e) {
            tvTaskHex.setText("加密失败: " + e.getMessage());
        }
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
        String u = etUrl.getText().toString().trim();
        if (u.isEmpty()) u = sp.getString("url", "");
        if (u.isEmpty()) throw new IllegalStateException("先填云端地址");
        return new Api(u, etToken.getText().toString().trim());
    }

    private void testConn() {
        final String url = etUrl.getText().toString().trim();
        setStatus("连接中…", 0xFFE0B341);
        pool.execute(() -> {
            try {
                Api a = new Api(url, etToken.getText().toString().trim());
                String s = a.stats();
                JSONObject o = new JSONObject(s);
                String dev = "?";
                try {
                    JSONArray d = o.getJSONArray("devices");
                    if (d.length() > 0) dev = d.getJSONObject(0).optString("name", "?");
                } catch (Exception ignore) {}
                log("连接成功 · " + dev);
                setStatus("在线", Ui.ACCENT);
                String objectInfo = a.get("/object_info/EncryptText");
                JSONObject oi = new JSONObject(objectInfo);
                if (!oi.has("EncryptText")) log("警告：云端缺少加密节点 EncryptText");
                else log("加密节点已就绪");
            } catch (Exception e) {
                setStatus("失败", 0xFFE05B5B);
                log("连接失败: " + e.getMessage());
            }
        });
    }

    private void generate() {
        cancelled.set(false);
        btnRun.setEnabled(false);
        btnRun.setText("生成中…");
        final String prompt = etPrompt.getText().toString().trim();
        if (prompt.isEmpty()) {
            toastLog("任务为空");
            resetBtn();
            return;
        }
        sp.edit().putString("prompt", prompt)
                .putString("steps", etSteps.getText().toString())
                .putString("w", etW.getText().toString())
                .putString("h", etH.getText().toString())
                .putString("url", etUrl.getText().toString().trim()).apply();

        final int steps = parseInt(etSteps.getText().toString(), 20);
        final int w = align(parseInt(etW.getText().toString(), 1024));
        final int h = align(parseInt(etH.getText().toString(), 1024));
        final String neg = etNeg.getText().toString().trim();

        setStatus("提交中…", 0xFFE0B341);
        log("— 新任务 —");

        pool.execute(() -> {
            try {
                Api a = api();
                // 1) 本地加密：明文不上网络
                byte[] blob = Crypto.encodeTask(prompt, key());
                String hexHead = Crypto.hex(blob).substring(0, 32);
                log("① 本地加密完成 " + blob.length + "B · " + hexHead + "…");

                // 2) 让云端加密节点留档一份密文（ComfyUI 原生节点）
                try {
                    JSONObject enc = new JSONObject();
                    enc.put("class_type", "EncryptText");
                    JSONObject ei = new JSONObject();
                    ei.put("text", prompt);
                    ei.put("seed", SEED);
                    ei.put("label", "1");
                    enc.put("inputs", ei);
                    JSONObject wf = new JSONObject();
                    wf.put("90", enc);
                    String pid = a.submit(wf);
                    log("② 云端加密留档 prompt_id=" + pid.substring(0, 8));
                } catch (Exception e) {
                    log("② 云端留档跳过: " + e.getMessage());
                }

                // 3) 提交出图任务
                JSONObject wf = Workflow.build(prompt, neg, w, h, steps,
                        Math.abs(new java.util.Random().nextLong() % 1_000_000_000L), 1.0,
                        DEF_UNET, DEF_CLIP, DEF_VAE, "apk");
                afterSubmit(a.submit(wf), a, blob);

            } catch (Exception e) {
                setStatus("失败", 0xFFE05B5B);
                log("出错: " + e.getMessage());
                resetBtn();
            }
        });
    }

    private void afterSubmit(String pid, Api a, byte[] blob) throws Exception {
        ui.post(() -> setStatus("生成中…", 0xFFE0B341));
        log("③ 已入队 " + pid.substring(0, 8) + "，等待 GPU…");
        JSONObject hist = a.wait(pid, 900_000, new Api.Progress() {
            public void tick(long sec) {
                final long s = sec;
                ui.post(() -> {
                    setStatus("生成中 " + s + "s", 0xFFE0B341);
                    btnRun.setText("生成中 " + s + "s");
                });
            }
            public boolean cancelled() { return cancelled.get(); }
        });

        // 4) 取回图片 -> 端上解密回原文，验证加密链路闭环
        JSONArray imgs = Api.images(hist);
        if (imgs.length() == 0) throw new IllegalStateException("没有返回图片");
        JSONObject im = imgs.getJSONObject(0);
        byte[] raw = a.view(im.optString("filename"), im.optString("subfolder"), im.optString("type", "output"));
        Bitmap bmp = BitmapFactory.decodeByteArray(raw, 0, raw.length);
        if (bmp == null) throw new IllegalStateException("图片解码失败");
        String echo = Crypto.decodeTask(blob, key());
        log("④ 出图成功 " + bmp.getWidth() + "x" + bmp.getHeight() + " · 本地解密校验: " + echo);
        ui.post(() -> {
            results.add(0, bmp);
            preview.setImageBitmap(bmp);
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
            btnRun.setText("一键生成");
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

    private static int align(int v) {
        int c = Math.max(256, Math.min(2048, v));
        return (c / 64) * 64;
    }

    @Override
    protected void onDestroy() {
        cancelled.set(true);
        pool.shutdownNow();
        super.onDestroy();
    }
}
