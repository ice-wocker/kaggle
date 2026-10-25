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
        this.user = user == null ? "" : user.trim();
        this.key = normalizeKey(key);
    }

    /** 用户名是 optString 出来的，null 就是 ""，不会真有 "null" 这种账号。 */
    private static boolean blank(String s) {
        return s == null || s.trim().isEmpty() || "null".equals(s);
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
        // 用户名还没探出来就先只发 key，Basic 的账号位留空：Kaggle 按 key 认人。
        // 别在这里拿 "me" 当账号发上去——hello 自己都不认这个名字。
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

    // ---------------- 凭据 ----------------

    /**
     * 新版 Kaggle 只发一个 KGAT_ 令牌，用户名也跟着它走；旧版才是 user + 32 位 key。
     * 两种情况都要能活，不能因为没填用户名就把人拦在门外。
     */
    static boolean isToken(String key) {
        return key != null && key.trim().toUpperCase().startsWith("KGAT_");
    }

    /** 从用户粘进来的任何东西里捞出真实凭据：带前缀的 URL、export 行、~/.kaggle/access_token 都有人贴。 */
    static String normalizeKey(String raw) {
        if (raw == null) return "";
        String s = raw.replace("\r", " ").replace("\n", " ").replace("\"", "").replace("'", "").trim();
        // export KAGGLE_API_TOKEN=KGAT_xxx / KAGGLE_KEY=xxx 这类整行；用户名不算凭据，不能抓错
        java.util.regex.Matcher kv = Pattern.compile("(?i)KAGGLE_(?:API_)?(?:TOKEN|KEY)\\s*=\\s*(\\S+)").matcher(s);
        if (kv.find()) s = kv.group(1);
        // 一整行 URL / 完整网页都贴进来的，取最长的那段合法字符。
        // 但 KGAT_ 令牌里带下划线，必须整段连着切，不能拆散前缀。
        java.util.regex.Matcher tok = Pattern.compile("(?i)KGAT_[A-Za-z0-9_\\-]{8,}|[A-Za-z0-9_\\-]{16,}").matcher(s);
        String best = "";
        while (tok.find()) if (tok.group().length() > best.length()) best = tok.group();
        return best.isEmpty() ? s : best;
    }

    /** hello 会回 token 与 userName，两个都拿，谁先到用谁。它说的错话不算数，见 probeOwner。 */
    JSONObject whoami() throws Exception {
        Res r = json(HOST + "/api/v1/hello", "GET", null);
        if (r.code == 401 || r.code == 403) throw new Api.AuthError("hello 不认这个 Key（HTTP " + r.code + "）");
        if (r.code >= 400) throw new Api.KaggleException("hello HTTP " + r.code);
        return r.json();
    }

    /**
     * hello 只是个探测点，不是判官。
     *
     * 它曾经把格式完全正确的 Key 判成 401，于是整条链路在第一步就死了——
     * 用户手上的东西没错，错的是我们太信这一句。现在这里只回答"探到没有"，
     * 401/403/404 一律当成"探不到"，绝不再据此对人说 Key 无效。
     */
    String probeOwner() {
        try {
            JSONObject o = whoami();
            String u = o.optString("userName", "");
            if (blank(u)) u = o.optString("username", "");
            if (blank(u)) u = o.optString("user", "");
            return blank(u) ? "" : u.trim();
        } catch (Exception e) {
            return "";
        }
    }

    /** 真实账号名：Key 属于谁，内核就推到谁名下。省掉一个要用户手填的字段。 */
    String resolveUser() throws Exception {
        String u = probeOwner();
        if (!blank(u)) return u;
        // 探不到就信用户：填了就用，没填就用 Kaggle 自己的 me
        if (!blank(user) && !"me".equals(user)) return user;
        return "me";
    }

    /**
     * 本地体检，不联网、不下判断：只回答"这一格看起来像不像凭据"。
     * 真伪交给 Kaggle，我们不再抢着说"你填错了"。
     */
    static String localHint(String raw) {
        String k = normalizeKey(raw);
        if (k.isEmpty()) return "还没填 Key";
        if (isToken(k)) return "看起来像一个 KGAT_ 令牌";
        if (k.length() == 32) return "看起来像旧式 32 位 key（用户名可留空）";
        return "长度 " + k.length() + "，既不像 KGAT_ 令牌也不像 32 位 key——先按原样试，不行再回 Kaggle 设置页核对";
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
        if (r.code == 401 || r.code == 403)
            throw new Api.AuthError("推内核被拒（HTTP " + r.code + "）：Key 没权限、已失效，或账号 " + owner
                    + " 不对（用户名可留空，留空就用 Kaggle 的 me）");
        if (r.code >= 400)
            throw new Api.KaggleException("push 失败 HTTP " + r.code + ": " + shortBody(r.text()) + hint(r.text()));
        return r.text();
    }

    // ---------------- 读状态 ----------------

    static final class KernelState {
        String status = "";
        String failure = "";
        boolean ok() { return "COMPLETE".equalsIgnoreCase(status); }
        /** Kaggle 有的内核是 COMPLETE 收尾，失败信息只在 failureMessage 里，所以两个都要看。 */
        boolean failed() {
            if (status != null && status.toUpperCase().contains("ERROR")) return true;
            return failure != null && !failure.isEmpty();
        }
        boolean running() { return "RUNNING".equalsIgnoreCase(status); }
        boolean queued() { return "QUEUED".equalsIgnoreCase(status); }
    }

    KernelState status(String owner) throws Exception {
        Res r = json(HOST + "/api/v1/kernels/status/" + enc(owner + "/" + SLUG), "GET", null);
        if (r.code == 404) return new KernelState();
        if (r.code == 401 || r.code == 403)
            throw new Api.AuthError("查内核状态被拒（HTTP " + r.code + "）：Key 与账号 " + owner
                    + " 不匹配——多半是用户名填错了，留空即可");
        if (r.code >= 400) throw new Api.KaggleException("status HTTP " + r.code);
        JSONObject o = r.json();
        KernelState k = new KernelState();
        // 老版本字段叫 status，新版本是 result.status（或 history[].status）
        k.status = o.optString("status", "").toUpperCase();
        if (k.status.isEmpty()) {
            JSONObject res = o.optJSONObject("result");
            if (res != null) k.status = res.optString("status", "").toUpperCase();
        }
        if (k.status.isEmpty()) {
            JSONArray hist = o.optJSONArray("history");
            if (hist != null && hist.length() > 0) {
                JSONObject last = hist.optJSONObject(hist.length() - 1);
                if (last != null) k.status = last.optString("status", "").toUpperCase();
            }
        }
        k.failure = o.optString("failureMessage", "");
        if (k.failure.isEmpty()) {
            JSONObject res = o.optJSONObject("result");
            if (res != null) k.failure = res.optString("failureMessage", "");
        }
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

    /** Kaggle 的报错常常只有一句 "Forbidden"，把最可能的下一步直接写出来。 */
    private static String hint(String body) {
        String b = body == null ? "" : body.toLowerCase();
        if (b.contains("phone") || b.contains("verif")) return "（账号需要先完成手机验证）";
        if (b.contains("gpu") || b.contains("quota")) return "（GPU 额度用完，或该账号还没有 GPU 资格）";
        if (b.contains("permission") || b.contains("forbidden")) return "（这个 Key 无权 push 到该账号，检查是否属同一账号）";
        return "";
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

    /** 手机里已有的地址+令牌还有效吗？查一次 /qwen/health：能连上且令牌对，就不用惊动 Kaggle。 */
    static boolean alive(String url, String token) {
        if (url == null || url.isEmpty()) return false;
        try {
            JSONObject h = probe(url, token);
            return h.optBoolean("ok") && h.optBoolean("auth", true);
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 端到端：起活 -> 等地址 -> 探活。返回可用的 {url, token}。
     * 每一步都上报，手机端能看到卡在哪，而不是干等一个转圈。
     */
    Map<String, String> bringUp(String apiKey, Progress p, long timeoutMs) throws Exception {
        long t0 = System.currentTimeMillis();
        String owner = resolveUser();
        // hello 探不出名字时 owner 是 me：照样往下走，别在门口把人拦住
        p.stage("me".equals(owner) ? "账号按 Key 自动识别" : "账号 " + owner);

        p.stage("检查云端会话");
        KernelState cur = status(owner);
        if (cur.running() || cur.queued()) {
            // 已经在跑/排着队就别再 push：push 会重启会话，等于白白再等一次冷启动
            p.stage(cur.queued() ? "会话排队中" : "会话已在运行");
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
            if (k.failed()) throw new Api.KaggleException(
                    "云端失败" + (k.failure.isEmpty() ? "(" + k.status + ")" : ": " + k.failure));

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
