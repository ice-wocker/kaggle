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

    // ---- 终端：输 Key 之后直接进「创建」，边生成边看 ----
    /** 三页：连接 / 创建 / 终端。终端是流式回显，创建页是那一屏参数 + 生成按钮。 */
    private LinearLayout pageConnect, pageCreate, pageTerminal;
    private TextView tvSteps, tvStageLine, tvPreviewNote, tvTermHead, tvEnvelope, tvTapHint;
    private LinearLayout stageChips;
    private ImageView termPreview;
    private android.widget.ProgressBar progBar;
    private android.webkit.WebView mirror;
    private Button btnTermPreview, btnTermMirror, btnTermCloud;
    private boolean mirrorMode = true;      // 终端里默认显示云端那个页面（与笔记本同一份）
    private boolean termPreviewMode = false; // 或本地直连 /qwen/preview 的过程图
    private String liveTaskId = "";
    private long liveT0 = 0;
    private int liveShown = 0;
    /** 终端那条流是否在跑：进终端页开，销毁时关。 */
    private final java.util.concurrent.atomic.AtomicBoolean liveOn =
            new java.util.concurrent.atomic.AtomicBoolean(false);
    private ImageView preview;
    private Button btnRun, btnStyle, btnSave, btnBatch, btnEnter;
    private int batch = 1;
    private android.widget.CompoundButton swNsfw, swQuality;
    private String styleName = "自动";
    private byte[] lastPng;
    private final StringBuilder logBuf = new StringBuilder();
    /** 当前请求打到的是云端（ComfyUI）还是 Kaggle：两者都回 401，但原因完全不同。 */
    private final java.util.concurrent.atomic.AtomicBoolean isCloudStep =
            new java.util.concurrent.atomic.AtomicBoolean(false);

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
    /**
     * 一屏三页：连接 → 创建 → 终端。
     *
     * 之前是一整条长滚动条，参数、结果、日志摞在一起，手机上要滑半天才知道
     * 「现在在干什么」。终端本该是「一眼看到它在动」，所以拆成页：
     * 填 Key 的那一页填完就走，创建页一屏装下所有能调的，终端页只留过程。
     */
    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(Ui.BG);
        LinearLayout root = Ui.column(this);
        int pad = Ui.dp(this, 16);
        root.setPadding(pad, pad, pad, Ui.dp(this, 32));
        scroll.addView(root);

        // Header
        LinearLayout head = Ui.column(this);
        TextView title = Ui.label(this, "Qwen-Image 2.1 · 终端", 22, Ui.TEXT);
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
        TextView sub = Ui.label(this, "填 Key 直接进创建 · 全过程实时回显 · 与笔记本同一份数据", 12, Ui.MUTED);
        Ui.add(head, sub, 4);
        Ui.add(root, head, 0);

        pageConnect = Ui.column(this);
        pageCreate = Ui.column(this);
        pageTerminal = Ui.column(this);
        Ui.add(root, pageConnect, 14);
        Ui.add(root, pageCreate, 14);
        Ui.add(root, pageTerminal, 14);

        buildConnectPage();
        buildCreatePage();
        buildTerminalPage();

        // 有 Key 就是老用户：开机直接落在创建页，不再让人多按一次
        showPage(!Kaggle.normalizeKey(sp.getString("key", "")).isEmpty());
        return scroll;
    }

    // ---- 页 ① 连接 ----
    private void buildConnectPage() {
        LinearLayout conn = Ui.panel(this);
        conn.addView(Ui.card(this, "① 一个 Key，剩下全自动"));
        Ui.add(conn, Ui.label(this, "Key 只存在本机，仅发给 kaggle.com 与你自己启动的那台 GPU。"
                + "粘完点「进入」——不用起脚本、不用传内核，云端会自己到位。", 11, Ui.MUTED), 6);
        etUser = Ui.input(this, "Kaggle 用户名（可留空；填就填 kaggle.com/ 后那个 handle，不是昵称）",
                sp.getString("user", ""), false);
        Ui.add(conn, etUser, 10);
        etKey = Ui.input(this, "粘 KGAT_… 令牌（或旧版 KAGGLE_KEY）", sp.getString("key", ""), false);
        Ui.add(conn, etKey, 8);
        btnKeyMode = btn("Key 形式：" + KEY_LABELS[keyMode]);
        btnKeyMode.setOnClickListener(v -> pickKeyMode());
        Ui.add(conn, btnKeyMode, 8);
        tvCloud = Ui.label(this, "云端未启动", 12, Ui.MUTED);
        Ui.add(conn, tvCloud, 10);
        btnEnter = new Button(this);
        btnEnter.setText("进入 →");
        btnEnter.setTextSize(17);
        btnEnter.setTextColor(0xFF07100E);
        btnEnter.setAllCaps(false);
        btnEnter.setBackground(Ui.rounded(this, Ui.ACCENT, 14, 0));
        btnEnter.setOnClickListener(v -> enterCreate());
        Ui.add(conn, btnEnter, 14);
        Ui.add(pageConnect, conn, 0);
    }

    // ---- 页 ② 创建 ----
    private void buildCreatePage() {
        LinearLayout task = Ui.panel(this);
        LinearLayout titleRow = new LinearLayout(this);
        titleRow.setOrientation(LinearLayout.HORIZONTAL);
        titleRow.setGravity(Gravity.CENTER_VERTICAL);
        titleRow.addView(Ui.card(this, "② 创建"),
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Button btnBack = btn("换 Key");
        btnBack.setOnClickListener(v -> showPage(false));
        titleRow.addView(btnBack);
        task.addView(titleRow);

        etPrompt = Ui.input(this, "想要什么图？一句中文也行",
                sp.getString("prompt", D_PROMPT), true);
        Ui.add(task, etPrompt, 10);

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
        btnRun.setText("开始生成 →");
        btnRun.setTextSize(17);
        btnRun.setTextColor(0xFF07100E);
        btnRun.setAllCaps(false);
        btnRun.setBackground(Ui.rounded(this, Ui.ACCENT, 14, 0));
        btnRun.setOnClickListener(v -> {
            if (cancelled.get()) return;
            generate();
        });
        Ui.add(task, btnRun, 14);

        LinearLayout res = new LinearLayout(this);
        res.setOrientation(LinearLayout.HORIZONTAL);
        res.setGravity(Gravity.CENTER_VERTICAL);
        btnBatch = btn("一次 " + batch + " 张");
        btnBatch.setOnClickListener(v -> pickBatch());
        res.addView(btnBatch, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        btnSave = btn("保存到相册");
        btnSave.setOnClickListener(v -> saveToGallery());
        LinearLayout.LayoutParams sp2 = new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        sp2.leftMargin = Ui.dp(this, 8);
        res.addView(btnSave, sp2);
        Ui.add(task, res, 10);

        preview = new ImageView(this);
        preview.setAdjustViewBounds(true);
        preview.setScaleType(ImageView.ScaleType.FIT_CENTER);
        preview.setBackground(Ui.rounded(this, 0xFF0B0F14, 12, Ui.LINE));
        preview.setMinimumHeight(Ui.dp(this, 200));
        Ui.add(task, preview, 10);
        gallery = Ui.column(this);
        Ui.add(task, gallery, 10);
        Ui.add(pageCreate, task, 0);
    }

    // ---- 页 ③ 终端 ----
    private void buildTerminalPage() {
        LinearLayout term = Ui.panel(this);
        LinearLayout headRow = new LinearLayout(this);
        headRow.setOrientation(LinearLayout.HORIZONTAL);
        headRow.setGravity(Gravity.CENTER_VERTICAL);
        tvTermHead = Ui.label(this, "③ 终端 · 等待任务", 15, Ui.TEXT);
        tvTermHead.setTypeface(null, android.graphics.Typeface.BOLD);
        headRow.addView(tvTermHead, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Button btnNew = btn("＋ 新建");
        btnNew.setOnClickListener(v -> showPage(true));
        headRow.addView(btnNew);
        term.addView(headRow);

        // 阶段芯片：和笔记本里那张控制台是同一份 stages 数据
        stageChips = Ui.column(this);
        Ui.add(term, stageChips, 8);
        tvStageLine = Ui.label(this, "阶段 —", 11, Ui.MUTED);
        Ui.add(term, tvStageLine, 6);

        progBar = new android.widget.ProgressBar(this, null,
                android.R.attr.progressBarStyleHorizontal);
        progBar.setMax(1000);
        Ui.add(term, progBar, 8);
        tvSteps = Ui.label(this, "等待云端回话…", 12, Ui.TEXT);
        tvSteps.setTypeface(android.graphics.Typeface.MONOSPACE);
        Ui.add(term, tvSteps, 6);
        tvTapHint = Ui.label(this, "照常滑屏；点一下过程图可在「云端控制台 / 过程图」之间切换", 10, Ui.MUTED);
        Ui.add(term, tvTapHint, 4);

        termPreview = new ImageView(this);
        termPreview.setAdjustViewBounds(true);
        termPreview.setScaleType(ImageView.ScaleType.FIT_CENTER);
        termPreview.setBackground(Ui.rounded(this, 0xFF0B0F14, 12, Ui.LINE));
        termPreview.setMinimumHeight(Ui.dp(this, 180));
        termPreview.setOnClickListener(v -> toggleTermView());
        Ui.add(term, termPreview, 8);
        tvPreviewNote = Ui.label(this, "过程图还没有（云端每采一步会推一张）", 11, Ui.MUTED);
        Ui.add(term, tvPreviewNote, 6);

        LinearLayout viewRow = new LinearLayout(this);
        viewRow.setOrientation(LinearLayout.HORIZONTAL);
        btnTermCloud = btn("控制台");
        btnTermCloud.setOnClickListener(v -> { termPreviewMode = false; toggleTermView(); });
        btnTermPreview = btn("过程图");
        btnTermPreview.setOnClickListener(v -> { termPreviewMode = true; toggleTermView(); });
        LinearLayout.LayoutParams bp = new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        bp.rightMargin = Ui.dp(this, 8);
        viewRow.addView(btnTermCloud, bp);
        viewRow.addView(btnTermPreview, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Ui.add(term, viewRow, 8);

        // 云端那个页面（笔记本里开的是同一个）：两边读同一份数据，所以同步是天然的
        mirror = new android.webkit.WebView(this);
        mirror.getSettings().setJavaScriptEnabled(true);
        mirror.getSettings().setDomStorageEnabled(true);
        mirror.setBackgroundColor(0xFF070B0F);
        mirror.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this, 420)));
        mirror.setVisibility(View.GONE);
        Ui.add(term, mirror, 8);

        Ui.add(term, Ui.label(this, "日志", 12, Ui.MUTED), 12);
        Button btnCopyLog = btn("复制日志");
        btnCopyLog.setOnClickListener(v -> {
            android.content.ClipboardManager cm =
                    (android.content.ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            cm.setPrimaryClip(ClipData.newPlainText("qwen21-log", tvLog.getText().toString()));
            log("日志已复制到剪贴板");
        });
        Ui.add(term, btnCopyLog, 6);
        tvLog = Ui.label(this, "就绪", 12, Ui.MUTED);
        tvLog.setTypeface(android.graphics.Typeface.MONOSPACE);
        tvLog.setMovementMethod(new ScrollingMovementMethod());
        Ui.add(term, tvLog, 6);
        tvEnvelope = Ui.label(this, "", 10, Ui.MUTED);
        tvEnvelope.setTypeface(android.graphics.Typeface.MONOSPACE);
        Ui.add(term, tvEnvelope, 8);
        Ui.add(pageTerminal, term, 0);
    }

    /** 页切换。终端页不显示时把 WebView 停掉，别在后台空转烧电。 */
    private void showPage(boolean create) {
        pageConnect.setVisibility(create ? View.GONE : View.VISIBLE);
        pageCreate.setVisibility(create ? View.VISIBLE : View.GONE);
        pageTerminal.setVisibility(create ? View.GONE : View.VISIBLE);
        if (create && mirror != null) mirror.setVisibility(View.GONE);
        if (!create) {
            attachMirror();
            applyTermView();
        }
    }

    /** 终端里切「云端控制台 / 过程图」：一个是整体镜像，一个是原生直连。 */
    private void toggleTermView() {
        termPreviewMode = !termPreviewMode;
        applyTermView();
    }

    private void applyTermView() {
        if (mirror == null) return;
        mirrorMode = !termPreviewMode;
        mirror.setVisibility(mirrorMode ? View.VISIBLE : View.GONE);
        termPreview.setVisibility(mirrorMode ? View.GONE : View.VISIBLE);
        tvPreviewNote.setVisibility(mirrorMode ? View.GONE : View.VISIBLE);
        btnTermCloud.setBackground(Ui.rounded(this, mirrorMode ? Ui.ACCENT : 0xFF1F2732, 12, Ui.LINE));
        btnTermCloud.setTextColor(mirrorMode ? 0xFF07100E : Ui.TEXT);
        btnTermPreview.setBackground(Ui.rounded(this, mirrorMode ? 0xFF1F2732 : Ui.ACCENT, 12, Ui.LINE));
        btnTermPreview.setTextColor(mirrorMode ? Ui.TEXT : 0xFF07100E);
        if (mirrorMode) attachMirror();
    }

    /**
     * 把云端控制台塞进 WebView。地址与令牌每次都从 sp 现取：
     * 会话交班后隧道域名会变，硬编码一次就永久白屏。
     */
    private void attachMirror() {
        final String base = sp.getString("url", "");
        final String tok = sp.getString("token", "");
        if (base.isEmpty() || tok.isEmpty()) {
            tvPreviewNote.setVisibility(View.VISIBLE);
            tvPreviewNote.setText("云端还没起来，起来后这里会自己接上");
            if (mirror != null) mirror.setVisibility(View.GONE);
            return;
        }
        if (mirrorMode) {
            String want = base + "/qwen/console?token=" + Api.enc(tok);
            if (!want.equals(mirror.getTag())) {
                mirror.setTag(want);
                mirror.loadUrl(want);
                tvPreviewNote.setVisibility(View.VISIBLE);
                tvPreviewNote.setText("云端控制台 · " + base.replace("https://", "")
                        + "（笔记本里开的是同一个页面）");
            }
            mirror.setVisibility(View.VISIBLE);
            termPreview.setVisibility(View.GONE);
        }
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

    /**
     * 「输 Key 之后直接进创建界面」。
     *
     * 有 Key 就立刻翻页——不要让人对着一个按钮等云端握手三十秒，
     * 那一步的消息本来就是终端该说的事：翻过去，终端里一行一行冒出来。
     */
    private void enterCreate() {
        saveKeyFields();
        String k = Kaggle.normalizeKey(etKey.getText().toString());
        if (k.isEmpty()) {
            log("先填 Kaggle API Key（KGAT_ 开头那个令牌，粘 export 整行也行）");
            setStatus("缺 Key", 0xFFE05B5B);
            return;
        }
        showPage(true);
        setStatus("已进入", Ui.ACCENT);
        log("Key " + Kaggle.localHint(k));
        log("布局就绪：左创建、右终端（竖屏是上下两段）");
        liveReset();
        // 缓存里还有地址就先探一下：通的话连「拉起云端」这一步都省掉
        pool.execute(() -> {
            String base = sp.getString("url", "");
            if (!base.isEmpty() && Kaggle.alive(base, sp.getString("token", ""))) {
                log("云端在线 · " + base.replace("https://", ""));
                ui.post(() -> {
                    tvCloud.setText("云端在线 · " + base.replace("https://", ""));
                    setStatus("在线", Ui.ACCENT);
                    attachMirror();
                });
                startLive();
            } else {
                log("云端还没起，第一次点「开始生成」会自己拉起来（约 10-15 分钟）");
            }
        });
    }

    // ---------------- 终端实时回显 ----------------
    private void liveReset() {
        liveTaskId = "";
        liveT0 = 0;
        liveShown = 0;
        ui.post(() -> {
            if (tvSteps != null) tvSteps.setText("等待任务…");
            if (progBar != null) progBar.setProgress(0);
            if (tvTermHead != null) tvTermHead.setText("③ 终端 · 待命");
            if (stageChips != null) stageChips.removeAllViews();
            if (tvPreviewNote != null && !mirrorMode)
                tvPreviewNote.setText("过程图还没有（云端每采一步会推一张）");
        });
    }

    /** 进入终端页就开始流：不需要点任何按钮，生成中才真正有东西可看。 */
    private void startLive() {
        if (!liveOn.compareAndSet(false, true)) return;
        pool.execute(() -> {
            while (liveOn.get() && !isFinishing()) {
                try {
                    Api a = apiCached();
                    JSONObject s = new JSONObject(a.consoleJson());
                    renderLive(s);
                } catch (Exception ignore) {
                    // 云端没起来 / 隧道抖一下都算正常，不打断终端
                }
                try { Thread.sleep(1500); } catch (InterruptedException e) { return; }
            }
        });
    }

    /**
     * 一条流里有三种东西：阶段线（云端走到哪一站）、任务进度（第几步）、日志。
     * 阶段与进度画在顶部，日志继续往下滚，过程图单独一个 ImageView 直连。
     * 注意：日志不在这一路多画一遍——本机 log() 已经写过了，云端的行由
     * pollTask 里那一步补进同一条时间线，否则同一条日志会出现两份。
     */
    private void renderLive(JSONObject s) {
        final String stage = s.optString("stage", "");
        final JSONArray stages = s.optJSONArray("stages");
        final JSONObject t = s.optJSONObject("task");
        ui.post(() -> {
            if (stages != null && stageChips != null) {
                stageChips.removeAllViews();
                LinearLayout row = new LinearLayout(this);
                row.setOrientation(LinearLayout.HORIZONTAL);
                for (int i = 0; i < stages.length() && i < 8; i++) {
                    String n = stages.optJSONObject(i).optString("name", "");
                    TextView chip = Ui.label(this, n, 10,
                            n.equals(stage) ? 0xFF07100E : Ui.MUTED);
                    chip.setBackground(Ui.rounded(this, n.equals(stage) ? Ui.ACCENT : 0xFF141B24, 99, Ui.LINE));
                    chip.setPadding(Ui.dp(this, 8), Ui.dp(this, 2), Ui.dp(this, 8), Ui.dp(this, 2));
                    LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
                    lp.rightMargin = Ui.dp(this, 6);
                    row.addView(chip, lp);
                }
                stageChips.addView(row);
            }
            if (tvStageLine != null && !stage.isEmpty()) tvStageLine.setText("阶段 " + stage);
            if (t == null || tvSteps == null) return;
            String st = t.optString("state", "");
            int total = t.optInt("total", 0), step = t.optInt("step", 0);
            double pct = t.optDouble("pct", 0);
            if ("running".equals(st)) {
                tvSteps.setText(String.format(java.util.Locale.US,
                        "%d/%d 步 · %.0f%% · %.0fs · %s", step, total, pct,
                        t.optDouble("elapsed", 0), t.optString("shape", "")));
                if (progBar != null) progBar.setProgress((int) Math.round(pct * 10));
                tvTermHead.setText("③ 终端 · 生成中");
                if (!mirrorMode) pullPreview(t.optString("id", ""), step);
            } else if ("done".equals(st)) {
                if (liveOn.get()) ui.post(() -> {
                    if (progBar != null) progBar.setProgress(1000);
                });
            } else if ("error".equals(st)) {
                String d = t.optString("detail", "");
                tvSteps.setText("云端报错：" + d.substring(0, Math.min(80, d.length())));
                tvTermHead.setText("③ 终端 · 失败");
            } else {
                tvTermHead.setText("③ 终端 · " + (stage.isEmpty() ? "待命" : stage));
            }
        });
    }

    /**
     * 过程图直连：/qwen/preview 每次回一张当前 latent 的缩略图。
     *
     * 为什么要自己拉而不全靠镜像：WebView 里那张小图在弱网下会糊成灰块，
     * 直连这条能自己控制频率（每 3 步一次），也让终端页在镜像关掉时仍有东西可看。
     */
    private void pullPreview(final String tid, final int step) {
        if (tid == null || tid.isEmpty()) return;
        if (step - liveShown < 3) return;   // 每 3 步拉一次，别把隧道挤爆
        liveShown = step;
        pool.execute(() -> {
            try {
                byte[] png = apiCached().preview(tid);
                if (png == null || png.length < 100) return;
                Bitmap bm = BitmapFactory.decodeByteArray(png, 0, png.length);
                if (bm == null) return;
                ui.post(() -> {
                    if (mirrorMode) return;
                    termPreview.setImageBitmap(bm);
                    tvPreviewNote.setText("第 " + step + " 步的过程图 · 采样中会脏，正常");
                });
            } catch (Exception ignore) {
                // 这一步图没拿到不影响任何事，下一步还有
            }
        });
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
            // 创建页右边那行小字：每次改参数就重算一遍，让人看见「发出去的是什么」
            if (tvEnvelope != null) {
                byte[] env = Crypto.packTask(fieldsFor(full, stepsNow(), cfgNow(), seedNow(), batch,
                        wNow(), hNow(), nsfwNow()), key());
                tvEnvelope.setText("信封 " + env.length + "B · " + Crypto.hex(env).substring(0, 24) + "…");
            }
        } catch (Exception e) {
            tvTaskHex.setText("加密失败: " + e.getMessage());
        }
    }

    // 创建页的那几个数：renderTasks 只是回显，读不到就退回默认值，别抛
    private int stepsNow() { return parseInt(etSteps == null ? "" : etSteps.getText().toString(), D_STEPS); }
    private double cfgNow() { return parseD(etCfg == null ? "" : etCfg.getText().toString(), D_CFG); }
    private long seedNow() { return parseLong(etSeed == null ? "" : etSeed.getText().toString(), D_SEED); }
    private int wNow() { return align(parseInt(etW == null ? "" : etW.getText().toString(), D_SIZE)); }
    private int hNow() { return align(parseInt(etH == null ? "" : etH.getText().toString(), D_SIZE)); }
    private boolean nsfwNow() { return swNsfw == null || swNsfw.isChecked(); }

    /** 一封待发的任务，字段顺序与 generate() 里那份一致：同一个界面切来切去，看到的必须是同一封信。 */
    private java.util.LinkedHashMap<String, Object> fieldsFor(String prompt, int steps, double cfg,
                                                             long seed, int batch, int w, int h, boolean nsfw) {
        java.util.LinkedHashMap<String, Object> f = new java.util.LinkedHashMap<>();
        f.put("prompt", prompt);
        f.put("negative", "");
        f.put("steps", steps);
        f.put("cfg", cfg);
        f.put("seed", seed);
        f.put("batch", batch);
        f.put("width", w);
        f.put("height", h);
        f.put("nsfw", nsfw);
        return f;
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

    /**
     * 打一行日志。同步写进内存与文件：手机断连、进程被杀之后还能从
     * logcat / 文件里翻回卡在哪一步（之前点了按钮一片空白，就是因为只往
     * 一个 TextView 里塞，后台线程抛异常时连异常都没人接）。
     */
    private void log(String s) {
        String line = String.format(java.util.Locale.US, "%s  %s", clock(), s);
        logBuf.append(line).append('\n');
        if (logBuf.length() > 60000) logBuf.delete(0, logBuf.length() - 40000);
        try {
            java.io.FileWriter fw = new java.io.FileWriter(logFile(), true);
            fw.write(line);
            fw.write('\n');
            fw.close();
        } catch (Exception ignore) {}
        ui.post(() -> tvLog.setText(logBuf.toString()));
    }

    private static String brief(Exception e) {
        String m = e.getMessage() == null ? "" : e.getMessage().replace('\n', ' ');
        return m.length() > 90 ? m.substring(0, 90) + "…" : e.getClass().getSimpleName() + " " + m;
    }

    private static String clock() {
        return new java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US)
                .format(new java.util.Date());
    }

    private java.io.File logFile() {
        return new java.io.File(getFilesDir(), "run.log");
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
        ensureCloud(false);
    }

    /**
     * @param force 上次出图被云端 401 挡下时置真：不信本地缓存的地址与令牌，
     *              重新走一遍「读日志拿地址」的握手。Key 本身没问题，别把它也一起否了。
     */
    private void ensureCloud(boolean force) throws Exception {
        String base = force ? "" : sp.getString("url", "");
        String token = sp.getString("token", "");
        if (force) log("上次被云端挡下，清掉缓存地址重新握手");
        if (!base.isEmpty()) {
            log("① 探活缓存地址 " + base.replace("https://", "") + " …（最多 15 秒）");
            try {
                // 缓存里的令牌可能是旧会话的：让 handshake 校准，别为了过期令牌去惊动 Kaggle
                final String good = Kaggle.handshake(base, token);
                if (!good.equals(token)) {
                    log("   令牌已按云端现时的校准，不用重填 Key");
                    sp.edit().putString("token", good).apply();
                }
                log("   云端在线，直接出图");
                ui.post(() -> {
                    tvCloud.setText("云端在线 · " + base.replace("https://", ""));
                    setStatus("在线", Ui.ACCENT);
                });
                return;
            } catch (Exception e) {
                log("   旧地址失效（" + brief(e) + "），重新拉起云端（能连上就不用管 Key）");
            }
        }
        final String key = Kaggle.normalizeKey(etKey.getText().toString());
        final String usr = etUser.getText().toString().trim();
        if (key.isEmpty()) throw new IllegalStateException("先填 Kaggle API Key（KGAT_ 开头那个令牌，粘 export 整行也行）");
        // 只拦真正空的：用户名留不留空由 Kaggle 判断，不由我们替它先否掉一个填对了的人
        log("Key " + Kaggle.localHint(key));
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
        log("   要等多久：首次装环境 + 17GB 权重约 10-15 分钟；已在线只是校准时通常几秒");
        java.util.Map<String, String> got = new Kaggle(effUser, key).bringUp(key, p, 25 * 60 * 1000L);
        final String url = got.get("url"), tok = got.get("token");
        sp.edit().putString("url", url).putString("token", tok)
                .putString("owner", got.get("owner")).putString("egress", got.get("egress")).apply();
        ui.post(() -> {
            tvCloud.setText("云端在线 · " + url.replace("https://", ""));
            setStatus("在线", Ui.ACCENT);
        });
        log("云 · 地址已下发 " + url);
    }

    private void generate() {
        generate(false);
    }

    private void generate(boolean rehandshake) {
        cancelled.set(false);
        btnRun.setEnabled(false);
        btnRun.setText("准备中…");
        // 点下去就翻到终端：从这一秒开始，所有动静都在那一页里发生
        showPage(false);
        startLive();
        // 用户点下去的第一秒就要有回音：以前这里什么都不打，
        // 只在后台线程里碰 SharedPreferences / 探活，一卡就是一分钟的空屏。
        log("=== 点击「一键出图」 · " + clock() + " ===");
        log("Key " + Kaggle.localHint(etKey.getText().toString())
                + " · 缓存云端 " + (sp.getString("url", "").isEmpty() ? "无" : "有"));
        final String rawPrompt = etPrompt.getText().toString().trim();
        if (rawPrompt.isEmpty()) {
            log("任务为空，先写一句想要什么图");
            resetBtn();
            return;
        }
        saveKeyFields();
        setStatus("准备中…", 0xFFE0B341);
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

        pool.execute(() -> {
            final long t0 = System.currentTimeMillis();
            try {
                log("② 任务 " + w + "x" + h + " · " + steps + " 步 · cfg " + cfg + " · " + batch + " 张"
                        + (nsfw ? " · NSFW 开" : ""));
                // 1) 云端：Key 换地址，已经在线就秒过
                isCloudStep.set(false);
                ensureCloud(rehandshake);
                log("   云端就绪 +" + ((System.currentTimeMillis() - t0) / 1000) + "s");
                Api a = api();
                isCloudStep.set(true);

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
                log("③ 本地封装 " + env.length + "B · " + Crypto.hex(env).substring(0, 32) + "…");
                log("参数 " + w + "x" + h + " · " + steps + " 步 · cfg " + cfg + " · seed " + useSeed
                        + (nsfw ? " · NSFW 开" : ""));

                // 3) 加密提交
                String tid = a.submitSealed(env);
                log("④ 已入队 " + tid.substring(0, 8));
                afterSubmit(tid, a, env);

            } catch (Api.AuthError e) {
                String msg = e.getMessage() == null ? "401" : e.getMessage();
                boolean fromCloud = msg.contains("/qwen/") || msg.contains("unauthorized")
                        || isCloudStep.get();
                log("出错: " + msg);
                if (fromCloud) {
                    setStatus("云端 401", 0xFFE05B5B);
                    log("  云端不认这个令牌，跟你的 Kaggle Key 无关");
                    if (!rehandshake && !cancelled.get()) {
                        log("  自动重来一次：先校准令牌，不行再丢掉旧地址重新握手");
                        resetBtn();
                        new android.os.Handler(android.os.Looper.getMainLooper())
                                .postDelayed(() -> generate(true), 1200);
                    } else {
                        log("  诊断: 云端会话可能刚交班，稍等几分钟再点一次；仍不行就把上面这行发我");
                        resetBtn();
                    }
                } else {
                    setStatus("Kaggle 拒绝", 0xFFE05B5B);
                    String tip = Kaggle.explain(etKey.getText().toString(),
                            etUser.getText().toString(), msg);
                    if (!tip.isEmpty()) log("  诊断: " + tip);
                    resetBtn();
                }
            } catch (InterruptedException ie) {
                log("已取消");
                resetBtn();
            } catch (Exception e) {
                setStatus("失败", 0xFFE05B5B);
                log("出错: " + e.getClass().getSimpleName() + (e.getMessage() == null ? "" : ": " + e.getMessage()));
                log("把这段日志（右上「复制日志」）发到 Issue 里，能直接定位到卡住的那一步");
                resetBtn();
            } finally {
                log("本次耗时 " + ((System.currentTimeMillis() - t0) / 1000) + "s");
            }
        });
    }

    /**
     * 等图这一段，以前是「每 15 秒报一行」，用户只能看秒数涨。
     * 现在两个口同时开：/qwen/task 说状态，/qwen/steps 说第几步，
     * /qwen/preview 说「现在长什么样」。三样一起，才叫实时看到生成到了哪步。
     */
    private void afterSubmit(String tid, Api a, byte[] env) throws Exception {
        ui.post(() -> {
            setStatus("生成中…", 0xFFE0B341);
            btnRun.setText("生成中…");
        });
        log("⑤ 已交给 GPU：T4 上 1024×1024/20 步约 4 分钟，每 2 秒刷一次进度");
        long t0 = System.currentTimeMillis();
        liveT0 = t0;
        liveTaskId = tid;
        liveShown = 0;
        JSONObject st = null;
        int fails = 0;
        int lastStep = -1;
        String lastState = "";
        String lastTip = "";
        long lastLogAt = 0;
        while (true) {
            if (cancelled.get()) throw new InterruptedException("已取消");
            try {
                st = a.task(tid);
                fails = 0;
            } catch (Exception e) {
                // 隧道抖动很常见，别一次读失败就把整次出图判死
                fails++;
                log("   轮询失败 " + fails + "/8 · " + e.getClass().getSimpleName()
                        + (e.getMessage() == null ? "" : ": " + e.getMessage()));
                if (fails >= 8) throw e;
                Thread.sleep(4000);
                continue;
            }
            String state = st.optString("state");
            String tip = st.optString("tip", "");
            if (!tip.isEmpty() && !tip.equals(lastTip)) {
                lastTip = tip;
                log("   云端提示: " + tip);
            }
            if ("done".equals(state)) break;
            if ("error".equals(state)) throw new IllegalStateException("云端报错: " + st.optString("detail"));

            long sec = (System.currentTimeMillis() - t0) / 1000;
            // 逐节点步骤：这是「生成到了哪一步」最直接的答案
            int step = -1, total = 0;
            JSONObject rep = null;
            try {
                rep = a.steps(tid);
                step = rep.optInt("step", -1);
                total = rep.optInt("total", 0);
            } catch (Exception ignore) {}
            if (step >= 0 && step != lastStep) {
                lastStep = step;
                final int fs = step, ft = total;
                final double pct = rep.optDouble("pct", 0);
                if (step > 0 || lastState.isEmpty()) {
                    log("   采样 " + step + "/" + (total > 0 ? String.valueOf(total) : "?") + " 步"
                            + " · " + sec + "s" + (pct > 0 ? String.format(java.util.Locale.US, " · %.0f%%", pct) : ""));
                }
                ui.post(() -> {
                    if (tvSteps != null)
                        tvSteps.setText(String.format(java.util.Locale.US, "%d/%d 步 · %.0f%% · %ds",
                                fs, ft, pct, sec));
                    if (progBar != null) progBar.setProgress((int) Math.round(pct * 10));
                    if (tvTermHead != null) tvTermHead.setText("③ 终端 · 生成中");
                });
            }
            // 过程图：终端页正在看的时候才拉，每 3 步一张
            if (step > 0 && !mirrorMode) pullPreview(tid, step);

            if (!state.equals(lastState)) {
                if (!state.isEmpty()) log("   云端状态 " + state + " · " + sec + "s");
                lastState = state;
            }
            // 没有逐节点回调时（老 ComfyUI）退回每 20 秒一行，别让日志彻底静止
            if (sec - lastLogAt >= 20) {
                lastLogAt = sec;
                if (step <= 0)
                    log("   生成中 " + sec + "s · " + state
                            + (st.has("queue") ? " · 队首" + st.optInt("queue") + " 个" : ""));
            }
            final long fs = sec;
            final int fstep = step;
            ui.post(() -> {
                setStatus("生成中 " + fs + "s", 0xFFE0B341);
                btnRun.setText("生成中 " + fs + "s");
                if (fstep <= 0 && tvSteps != null) tvSteps.setText("等第一步 · " + fs + "s · " + state);
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
        final double usedSec = st.optDouble("elapsed", 0);
        log("⑥ 出图 " + got.size() + " 张 " + got.get(0).getWidth() + "x" + got.get(0).getHeight()
                + " · 用时 " + st.optDouble("elapsed", 0) + "s · 信封 " + env.length + "B（全程未明文过网）");
        ui.post(() -> {
            for (int i = got.size() - 1; i >= 0; i--) results.add(0, got.get(i));
            lastPng = png;
            preview.setImageBitmap(got.get(0));
            rebuildGallery();
            setStatus("完成", Ui.ACCENT);
            resetBtn();
            liveOn.set(false);
            if (tvTermHead != null) tvTermHead.setText("③ 终端 · 完成");
            if (tvSteps != null) tvSteps.setText("完成 · " + got.size() + " 张 · " + usedSec + "s");
            if (progBar != null) progBar.setProgress(1000);
            tvPreviewNote.setVisibility(View.VISIBLE);
            tvPreviewNote.setText("成品已存进「创建」页的结果区 · 过程图到此结束");
            // 图都出来了：终端那条流收掉，终端页留一个静态的「完成」
            termPreviewMode = false;
            applyTermView();
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
            btnRun.setText("开始生成 →");
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
                        attachMirror();
                    });
            });
        }
        // 回到前台就继续流：终端页上的数字不该停在那儿等人再点一下
        if (!sp.getString("url", "").isEmpty()) startLive();
    }

    @Override
    protected void onPause() {
        super.onPause();
        // 后台不拉流，也不让 WebView 里的定时器继续跑
        liveOn.set(false);
        if (mirror != null) mirror.onPause();
    }

    @Override
    protected void onDestroy() {
        cancelled.set(true);
        liveOn.set(false);
        if (mirror != null) {
            mirror.loadUrl("about:blank");
            mirror.destroy();
        }
        pool.shutdownNow();
        super.onDestroy();
    }
}
