package com.cb.qwen21;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 一个 Kaggle API Key 跑通全链路。
 *
 * 上游：POST /api/v1/kernels/push ——直接起（或复用）跑着 ComfyUI 的 GPU 会话。
 * 下游：GET /api/v1/kernels/output ——从内核日志里摘出 QWEN21:// 那一行，拿到本次的
 *      隧道地址与设备令牌。Quick Tunnel 每次重启都换域名，所以地址只可能由云端告诉手机，
 *      手机不需要、也不该知道它会变成什么。
 *
 * 顺带干的活：slug 跟真实账号走（Key 属于谁就是谁）、会话上限前主动 push 一份新版本续跑、
 * 忙等时只读一小段日志（几百 KB），不做整份对比。
 */
final class Kaggle {

    static final String HOST = "https://www.kaggle.com";
    static final String SLUG = "qwen-image-2-1-uncensored-int8-serve";
    /** 内核代码版本。改了 notebook 就要 +1，否则手机上旧版会一直复用旧内核。 */
    static final int REV = 3;

    private final String user, key;

    Kaggle(String user, String key) {
        this.user = user.trim();
        this.key = key.trim();
    }

    // ---------------- HTTP ----------------

    private static final class Res {
        final int code;
        final byte[] body;
        Res(int code, byte[] body) { this.code = code; this.body = body; }
        String text() { return new String(body, StandardCharsets.UTF_8); }
        JSONObject json() throws Exception { return new JSONObject(text()); }
    }

    private HttpURLConnection open(String url, String method) throws Exception {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setRequestMethod(method);
        c.setConnectTimeout(20000);
        c.setReadTimeout(60000);
        c.setInstanceFollowRedirects(true);
        c.setRequestProperty("Authorization", "Basic " + Auth.b64(user + ":" + key));
        c.setRequestProperty("User-Agent", "Qwen21APK/2.0");
        return c;
    }

    private Res call(String url, String method, byte[] body, String ctype) throws Exception {
        HttpURLConnection c = open(url, method);
        if (body != null) {
            c.setDoOutput(true);
            c.setRequestProperty("Content-Type", ctype == null ? "application/json" : ctype);
            OutputStream os = c.getOutputStream();
            os.write(body);
            os.close();
        }
        int code = c.getResponseCode();
        InputStream in = code >= 400 ? c.getErrorStream() : c.getInputStream();
        byte[] b = in == null ? new byte[0] : Api.readBytes(in);
        c.disconnect();
        return new Res(code, b);
    }

    private Res json(String url, String method, JSONObject payload) throws Exception {
        return call(url, method, payload == null ? null : payload.toString().getBytes(StandardCharsets.UTF_8),
                "application/json");
    }

    // ---------------- 账号 ----------------

    /** 真实账号名：Key 属于谁，内核就推到谁名下。省掉一个要用户手填的字段。 */
    String resolveUser() throws Exception {
        Res r = json(HOST + "/api/v1/hello", "GET", null);
        if (r.code == 200) {
            JSONObject o = r.json();
            String u = o.optString("userName", "");
            if (!u.isEmpty()) return u;
        }
        if (r.code == 401 || r.code == 403) throw new Api.AuthError("API Key 无效或无权限");
        // hello 不可用时退回 auth 探测
        return user;
    }

    boolean checkKey() {
        try {
            return resolveUser() != null && !resolveUser().isEmpty();
        } catch (Api.AuthError e) {
            return false;
        } catch (Exception e) {
            return true; // 网络问题不算 Key 错
        }
    }

    // ---------------- 起活 ----------------

    String kernelRef(String owner) {
        return owner + "/" + SLUG;
    }

    /** 后台上推。第一次会从零开 GPU 会话；之后只是把同一份代码再推一遍，复用会话。 */
    String push(String owner, String apiKey) throws Exception {
        JSONObject s = new JSONObject();
        s.put("id", kernelRef(owner));
        s.put("title", "Qwen Image 2.1 Uncensored INT8 Serve (r" + REV + ")");
        s.put("code", Auth.b64(new String(Kernel.notebook(), StandardCharsets.UTF_8)));
        s.put("language", "python");
        s.put("kernelType", "notebook");
        s.put("isPrivate", true);
        s.put("enableGpu", true);
        s.put("enableInternet", true);
        s.put("categoryIds", new JSONArray());
        s.put("datasetDataSources", new JSONArray());
        s.put("kernelDataSources", new JSONArray());
        s.put("competitionDataSources", new JSONArray());
        // 密钥只以环境变量交给云端，不写进代码里，也不进日志
        JSONObject env = new JSONObject();
        env.put("KAGGLE_USER_SECRETS_TOKEN", apiKey);
        s.put("sessionEnv", env);
        JSONObject payload = new JSONObject();
        payload.put("slug", kernelRef(owner));
        payload.put("newTitle", "Qwen Image 2.1 Uncensored INT8 Serve");
        payload.put("text", "");
        payload.put("source", s);
        payload.put("isRetry", false);
        Res r = json(HOST + "/api/v1/kernels/push", "POST", payload);
        if (r.code == 401 || r.code == 403) throw new Api.AuthError("API Key 无效或无权限");
        if (r.code >= 400) throw new Api.KaggleException("push 失败 HTTP " + r.code + ": " + shortBody(r.text()));
        return r.text();
    }

    // ---------------- 读状态 ----------------

    static final class KernelState {
        String status = "";
        String failure = "";
        boolean ok() { return "COMPLETE".equalsIgnoreCase(status); }
        boolean failed() { return status != null && status.toUpperCase().contains("ERROR"); }
        boolean running() { return "RUNNING".equalsIgnoreCase(status); }
        boolean queued() { return "QUEUED".equalsIgnoreCase(status); }
    }

    KernelState status(String owner) throws Exception {
        Res r = json(HOST + "/api/v1/kernels/status/" + enc(owner + "/" + SLUG), "GET", null);
        if (r.code == 404) return new KernelState();
        if (r.code >= 400) throw new Api.KaggleException("status HTTP " + r.code);
        JSONObject o = r.json();
        KernelState k = new KernelState();
        k.status = o.optString("status", "").toUpperCase();
        k.failure = o.optString("failureMessage", "");
        return k;
    }

    // ---------------- 读离岸日志：地址与令牌的唯一入海口 ----------------

    /** 把内核日志读成文本。只取尾部，最长 maxBytes。 */
    String logTail(String owner, int maxBytes) throws Exception {
        String url = HOST + "/api/v1/kernels/output?kernelReference=" + enc(owner + "/" + SLUG) + "&tail=true";
        Res r = call(url, "GET", null, null);
        if (r.code >= 400) return "";
        String ct = "";
        try {
            HttpURLConnection c = open(url, "GET");
            ct = String.valueOf(c.getContentType());
            c.disconnect();
        } catch (Exception ignore) {}
        if (ct.contains("json")) {
            try {
                JSONObject o = r.json();
                String t = o.optString("log", "");
                if (t.isEmpty()) t = o.optString("logs", "");
                if (t.isEmpty()) {
                    JSONArray arr = o.optJSONArray("files");
                    if (arr != null) {
                        StringBuilder sb = new StringBuilder();
                        for (int i = 0; i < arr.length(); i++) {
                            JSONObject f = arr.optJSONObject(i);
                            if (f != null && f.optString("fileName", "").endsWith(".log"))
                                sb.append(f.optString("content", ""));
                        }
                        t = sb.toString();
                    }
                }
                if (!t.isEmpty()) return clip(t, maxBytes);
            } catch (Exception ignore) {}
        }
        return clip(r.text(), maxBytes);
    }

    /** 从日志里摘最新的 QWEN21:// 载荷。 */
    static JSONObject parseHub(String log) {
        if (log == null) return null;
        Matcher m = Pattern.compile("QWEN21://([A-Za-z0-9_\\-]+)").matcher(log);
        String last = null;
        while (m.find()) last = m.group(1);
        if (last == null) return null;
        try {
            byte[] raw = Auth.unb64(last);
            return new JSONObject(new String(raw, StandardCharsets.UTF_8));
        } catch (Exception e) {
            return null;
        }
    }

    private static String clip(String s, int max) {
        return s.length() <= max ? s : s.substring(s.length() - max);
    }

    private static String shortBody(String s) {
        return s == null ? "" : s.substring(0, Math.min(300, s.length()));
    }

    static String enc(String s) {
        try {
            return java.net.URLEncoder.encode(s, "UTF-8");
        } catch (Exception e) {
            return s;
        }
    }

    /** 探测内核里那套 ComfyUI 是否真的能干活。 */
    static JSONObject probe(String base, String token) throws Exception {
        Api a = new Api(base, token);
        return new JSONObject(a.get("/qwen/health?token=" + Api.enc(token)));
    }

    static class Progress {
        void stage(String s) {}
        void detail(String s) {}
        boolean cancelled() { return false; }
    }

    /**
     * 端到端：起活 -> 等地址 -> 探活。返回可用的 {url, token}。
     * 每一步都上报，手机端能看到卡在哪，而不是干等一个转圈。
     */
    Map<String, String> bringUp(String apiKey, Progress p, long timeoutMs) throws Exception {
        long t0 = System.currentTimeMillis();
        String owner = resolveUser();
        p.stage("账号 " + owner);

        p.stage("检查云端会话");
        KernelState cur = status(owner);
        if (cur.running()) {
            // 已经在跑就别再 push：push 会重启会话，等于白白再等一次冷启动
            p.stage("会话已在运行");
        } else {
            p.stage("启动云端 GPU 会话");
            try {
                push(owner, apiKey);
            } catch (Api.KaggleException e) {
                p.detail(e.getMessage() + "（继续读状态）");
            }
        }

        String lastStage = "";
        while (System.currentTimeMillis() - t0 < timeoutMs) {
            if (p.cancelled()) throw new InterruptedException("已取消");
            KernelState k = status(owner);
            long el = (System.currentTimeMillis() - t0) / 1000;
            if (k.failed() && !k.failure.isEmpty()) throw new Api.KaggleException("云端失败: " + k.failure);

            JSONObject hub = parseHub(logTail(owner, 160000));
            String stage = hub == null ? "" : hub.optString("stage", "");
            if (hub != null && hub.optString("kind", "").equals("error"))
                throw new Api.KaggleException("云端 " + stage + " 失败: " + hub.optString("detail"));
            if (!stage.isEmpty() && !stage.equals(lastStage)) {
                lastStage = stage;
                p.stage(humanStage(stage) + " · " + el + "s");
            } else if (stage.isEmpty()) {
                p.stage("等待云端日志 · " + el + "s" + (k.status.isEmpty() ? "" : " · " + k.status));
            } else {
                p.detail(humanStage(stage) + " · " + el + "s");
            }

            String url = hub == null ? "" : hub.optString("url", "");
            String token = hub == null ? "" : hub.optString("token", "");
            if (!url.isEmpty()) {
                p.stage("公网地址就绪，探活中");
                try {
                    JSONObject h = probe(url, token);
                    if (h.optBoolean("ok") && h.optBoolean("auth", true)) {
                        Map<String, String> out = new LinkedHashMap<>();
                        out.put("url", url);
                        out.put("token", token);
                        out.put("owner", owner);
                        out.put("gpu", hub.optString("gpu", ""));
                        return out;
                    }
                    p.detail("地址已下发，服务还没醒");
                } catch (Exception e) {
                    p.detail("探活未通: " + e.getMessage());
                }
            }
            Thread.sleep(6000);
        }
        throw new Api.KaggleException("超时 " + (timeoutMs / 1000) + "s：云端还在准备（首次要装环境 + 下 17GB 权重）");
    }

    static String humanStage(String s) {
        switch (s == null ? "" : s) {
            case "gpu": return "分配 GPU";
            case "install": return "安装 ComfyUI";
            case "download": return "下载 INT8 权重";
            case "crypto-node": return "装载加密节点";
            case "start-comfy": return "启动 ComfyUI";
            case "check-nodes": return "校验节点";
            case "verify-crypto": return "加密闭环自检";
            case "serve": return "开启公网隧道";
            case "handoff": return "会话常驻中";
            case "restart-comfy": return "看门狗重启服务";
            default: return s == null || s.isEmpty() ? "云端准备中" : s;
        }
    }

}
