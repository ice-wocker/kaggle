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

    /** 老的 www 端点：hello / push / version 还活着。 */
    static final String HOST = "https://www.kaggle.com";
    /**
     * 新的 midtier：状态与日志只在这里。
     *
     * 这一条是「点了按钮一小时没反应」的真正原因。以前状态打 /api/v1/kernels/status/{owner}/{slug}
     * （也试过 ?kernelReference=…），这个路由在 www 上根本不存在，回的是 404 + 一页登录页 HTML；
     * 代码把 404 当成「内核不存在」→ 状态永远是空 → 每一轮都判成「不在线」→ 再 push 一次。
     * 也就是：请求发出去了，回的永远是一句「没有这个接口」，然后被当成「云端还没好」。
     */
    static final String MIDTIER = "https://api.kaggle.com";
    static final String SLUG = "qwen-image-2-1-uncensored-int8-serve";
    /** 内核代码版本。改了 notebook 就要 +1，否则手机上旧版会一直复用旧内核。 */
    static final int REV = 8;
    /** 与云端 qwen_setup.py 的 REV 同一份稳定令牌：云端换代码、手机没换包时也能进。 */
    static final String DEVICE_TOKEN = "QW21-REV8";

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

    /**
     * 凭据怎么带。Kaggle 对「旧式 key + Basic」和「KGAT_ 令牌」挑得不一样，
     * 401 时换下一个再试——总比让人回去重填一个字都没错的 Key 强。
     */
    private String[][] authModes() {
        if (isToken(key)) {
            return new String[][]{
                    {"Authorization", "Bearer " + key},
                    {"Authorization", "Basic " + Auth.b64(user + ":" + key)},
            };
        }
        return new String[][]{
                {"Authorization", "Basic " + Auth.b64(user + ":" + key)},
                {"Authorization", "Bearer " + key},
        };
    }

    /** 哪些接口已经试出「该换一种带法」，避免每次 401 都白跑一趟。 */
    private static final java.util.Set<String> AUTH_URLS = new java.util.HashSet<>();

    private static HttpURLConnection open(String url, String method, String[] mode) throws Exception {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setRequestMethod(method);
        c.setConnectTimeout(20000);
        c.setReadTimeout(60000);
        c.setInstanceFollowRedirects(true);
        // 用户名还没探出来就先只发凭据本身，Basic 的账号位留空：Kaggle 按 key 认人。
        // 别在这里拿 "me" 当账号发上去——hello 自己都不认这个名字。
        c.setRequestProperty(mode[0], mode[1]);
        c.setRequestProperty("User-Agent", "Qwen21APK/2.0");
        return c;
    }

    /** 每一条真正发出去的请求都记一行。用户看不到网络，只能看日志——那就把网络写在日志里。 */
    private void tap(String method, String url, int code) {
        Progress p = current;
        if (p != null) p.detail("· " + method + " " + shortUrl(url) + " → " + code);
    }

    private static String shortUrl(String url) {
        return url.replace("https://www.kaggle.com", "").replace("https://api.kaggle.com", "")
                .replace("/api/v1", "").replace("/v1/kernels.KernelsApiService", "");
    }

    /** 正在跑的那次「一键出图」：只为了把请求打到日志上，不做别的。 */
    private volatile Progress current;

    private Res call(String url, String method, byte[] body, String ctype) throws Exception {
        Res r = callOnce(url, method, body, ctype, authModes()[0]);
        tap(method, url, r.code);
        if (r.code == 401 || r.code == 403) {
            String[] alt = authModes()[1];
            Res r2 = callOnce(url, method, body, ctype, alt);
            tap(method + "(换带法)", url, r2.code);
            if (r2.code < 400) {
                // 换对了：记住这条接口的带法，下次直接用
                synchronized (AUTH_URLS) { AUTH_URLS.add(url); }
                return r2;
            }
            // 两种带法都被拒，把"看起来更像真的"那条报出去
            return r2;
        }
        return r;
    }

    private Res callOnce(String url, String method, byte[] body, String ctype, String[] mode) throws Exception {
        HttpURLConnection c = open(url, method, mode);
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

    /**
     * hello 会回 token 与 userName，两个都拿，谁先到用谁。
     * 注意它对「怎么带凭据」比别的端点挑：同一个 Key，push 能用、hello 回 401 是常事，
     * 所以这里绝不抛 AuthError —— 一旦抛了，后面真正干活的接口就没机会说话。
     */
    JSONObject whoami() throws Exception {
        Res r = json(HOST + "/api/v1/hello", "GET", null);
        if (r.code >= 400) return new JSONObject();
        try {
            return r.json();
        } catch (Exception e) {
            return new JSONObject();
        }
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
        if (isToken(k)) return "看起来像一个 KGAT_ 令牌（" + k.length() + " 位）";
        if (k.length() == 32) return "看起来像旧式 32 位 key（用户名可留空）";
        return "长度 " + k.length() + "，既不像 KGAT_ 令牌也不像 32 位 key——先按原样试，不行再回 Kaggle 设置页核对";
    }

    /**
     * 把失败拆成人能听懂的一句。401 不等于 Key 错：字段错、账号名对不上、
     * 令牌被网页截断，都会让 Kaggle 回同一个 401，而人只看得到"鉴权没过"。
     */
    static String explain(String rawKey, String user, String err) {
        String k = normalizeKey(rawKey);
        String u = user == null ? "" : user.trim();
        String e = err == null ? "" : err;
        // 两条 401 完全不是一回事，别再混成一句"鉴权没过"
        if (e.contains("云端网关") || e.contains("网关"))
            return "这是隧道/网关挡的，不是 Key：trycloudflare 出口 IP 轮换过，新 IP 不在白名单里；"
                    + "点重试会重新等云端下发地址";
        if (e.contains("hello") && !e.contains("推内核") && !e.contains("状态"))
            return "hello 只是探测点，它不认不代表 Key 错；真正干活的是 push/status，看那两步的报错";
        if (e.contains("推送") || e.contains("推内核") || e.contains("状态") || e.contains("Kaggle 拒绝"))
            return "这一步是 Kaggle 本人在说话：把用户名清空、只留 KGAT_ 令牌再试；"
                    + "还 401 就去 kaggle.com/settings 重新生成令牌";
        if (!isToken(k) && k.length() == 32 && u.isEmpty())
            return "旧式 32 位 key 要靠用户名配对，这一格填 kaggle.com/ 后面的 handle";
        if (u.chars().anyMatch(c -> c > 127)) return "用户名里有非 ASCII 字符：" + u + " 多半是昵称，换成 handle";
        return e;
    }

    // ---------------- 起活 ----------------

    String kernelRef(String owner) {
        return owner + "/" + SLUG;
    }

    /** 本次是否真的把 push 打出去了：日志里要说清楚「发出去了」还是「没发」。 */
    private boolean pushed = false;

    boolean pushed() { return pushed; }

    /** 后台上推。第一次会从零开 GPU 会话；之后只是把同一份代码再推一遍，复用会话。 */
    String push(String owner, String apiKey) throws Exception {
        JSONObject s = new JSONObject();
        s.put("id", kernelRef(owner));
        s.put("title", "Qwen Image 2.1 Uncensored INT8 Serve (r" + REV + ")");
        s.put("code", Auth.b64(Kernel.notebook()));
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
        pushed = r.code < 400;
        if (r.code == 401 || r.code == 403)
            throw new Api.AuthError("Kaggle 拒绝推内核 · HTTP " + r.code + " · 账号 " + owner
                    + "（这时才轮到说凭据：Key 失效 / 账号不是这个 Key 的 / 手机号没验证。"
                    + "用户名留空就用 Kaggle 的 me）");
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

    /**
     * 内核状态。只走 midtier 的 RPC：www 上那条 /api/v1/kernels/status/... 是不存在的路由，
     * 回 404 + HTML，任何「有没有在线」的判断都建立在它上面，所以整条链路从来没对过。
     */
    KernelState status(String owner) throws Exception {
        JSONObject req = new JSONObject();
        req.put("userName", owner);
        req.put("kernelSlug", SLUG);
        Res r = json(MIDTIER + "/v1/kernels.KernelsApiService/GetKernelSessionStatus", "POST", req);
        if (r.code == 401 || r.code == 403)
            throw new Api.AuthError("查内核状态被拒（HTTP " + r.code + "）：Key 与账号 " + owner
                    + " 不匹配——多半是用户名填错了，留空即可");
        if (r.code >= 400) throw new Api.KaggleException("status HTTP " + r.code + ": " + shortBody(r.text()));
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
        if ("null".equals(k.failure)) k.failure = "";
        if (k.failure.isEmpty()) {
            JSONObject res = o.optJSONObject("result");
            if (res != null) k.failure = res.optString("failureMessage", "");
        }
        if (k.status.isEmpty() && o.optJSONObject("error") != null)
            k.failure = o.optJSONObject("error").optString("message", "");
        return k;
    }

    // ---------------- 读离岸日志：地址与令牌的唯一入海口 ----------------

    /**
     * 把内核日志读成文本。只取尾部，最长 maxBytes。
     *
     * 端点也是换过的：www 上的 /api/v1/kernels/output 只认 userName + kernelSlug 两个查询参数，
     * 传 kernelReference 一律 400。返回体是 {"log": "<JSON 数组字符串>", "files": [...]}，
     * 里面每条是 {stream_name,time,data}——真内容在 data 里（实测 19.5KB 的 log 拼回
     * 10.3KB 文本、13 条 QWEN21://）。字段名是 log，不是 logNullable，
     * 猜错一次的代价是：端上永远觉得「日志是空的」。
     *
     * 以前这里把整个 JSON 当纯文本 clip，日志尾部 160KB 全是这些包装，
     * QWEN21:// 那一行不是被截断就是根本没进窗口，于是「日志有内容但没有地址」。
     */
    String logTail(String owner, int maxBytes) throws Exception {
        JSONObject req = new JSONObject();
        req.put("userName", owner);
        req.put("kernelSlug", SLUG);
        req.put("pageSize", 100);
        Res r = json(MIDTIER + "/v1/kernels.KernelsApiService/ListKernelSessionOutput", "POST", req);
        if (r.code >= 400) return "";
        JSONObject o = r.json();
        String raw = o.optString("log", "");
        if (raw.isEmpty() || "null".equals(raw)) raw = o.optString("logNullable", "");
        if (raw.isEmpty() || "null".equals(raw)) raw = o.optString("logs", "");
        StringBuilder sb = new StringBuilder(raw.length());
        try {
            JSONArray arr = new JSONArray(raw);
            for (int i = 0; i < arr.length(); i++) {
                JSONObject e = arr.optJSONObject(i);
                if (e != null) sb.append(e.optString("data", ""));
            }
        } catch (Exception e) {
            sb.append(raw);
        }
        String text = sb.length() > 0 ? sb.toString() : raw;
        return clip(text, maxBytes);
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

    /**
     * 探活并顺带把令牌校准好。返回的 token 是云端现时认的那个。
     *
     * 设备令牌有两种来源：内核日志里那条 ready（可能是上一会话留下的旧值）、
     * 隧道上的 /qwen_token.txt（服务端每次启动覆盖写入）。哪个能用就用哪个，
     * 都不对就退回内置的稳定令牌——它跟着 REV 走，云端认、本地也知道。
     * 拿不到"服务活着"才算失败：令牌过期不该让人以为整台机器没醒。
     */
    static String handshake(String base, String token) throws Exception {
        String[] cands = {token, DEVICE_TOKEN};
        if (token == null || token.isEmpty()) cands = new String[]{DEVICE_TOKEN};
        JSONObject last = null;
        for (String t : cands) {
            JSONObject h = probe(base, t);
            if (h.optBoolean("auth")) return t;
            last = h;
        }
        java.util.List<String> file = freshTokens(base);
        for (String t : file) {
            if (probe(base, t).optBoolean("auth")) return t;
        }
        if (last != null && !last.optBoolean("ok"))
            throw new IllegalStateException("隧道通了但服务没醒：" + base);
        // 服务活着、所有已知令牌都不认：照原样交出去，出图那一步会给出具体报错
        return token == null || token.isEmpty() ? DEVICE_TOKEN : token;
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
            // ok 现在专指"服务活着"，auth 才是令牌对不对；老云端只回 ok 时按老规矩读
            return h.optBoolean("ok") && h.optBoolean("auth", true);
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 拿云端文件里现成的令牌。只在「隧道通、令牌不对」时用：
     * 服务端会把当前令牌写在 /kaggle/working/qwen_token.txt，暴露在隧道根路径下。
     * 这样设备令牌轮换（会话交班）后手机不用等人重装 APK。
     */
    static java.util.List<String> freshTokens(String base) {
        java.util.List<String> out = new java.util.ArrayList<>();
        if (base == null || base.isEmpty()) return out;
        try {
            HttpURLConnection c = (HttpURLConnection) new URL(base + "/qwen_token.txt").openConnection();
            c.setConnectTimeout(10000);
            c.setReadTimeout(10000);
            c.setRequestProperty("User-Agent", "Qwen21APK/2.0");
            if (c.getResponseCode() >= 400) return out;
            String text = new String(Api.readBytes(c.getInputStream()), StandardCharsets.UTF_8);
            c.disconnect();
            for (String line : text.split("\\r?\\n")) {
                String t = line.trim();
                if (!t.isEmpty() && t.length() <= 200 && !out.contains(t)) out.add(t);
            }
        } catch (Exception ignore) {
        }
        return out;
    }

    /**
     * 端到端：起活 -> 等地址 -> 探活。返回可用的 {url, token}。
     * 每一步都上报，手机端能看到卡在哪，而不是干等一个转圈。
     */
    Map<String, String> bringUp(String apiKey, Progress p, long timeoutMs) throws Exception {
        long t0 = System.currentTimeMillis();
        current = p;
        String owner = resolveUser();
        // hello 探不出名字时 owner 是 me：照样往下走，别在门口把人拦住
        p.stage("me".equals(owner) ? "账号按 Key 自动识别" : "账号 " + owner);

        // 日志最多每 5 秒拉一次、内容没变就复用：首次那几个阶段每步好几分钟，
        // 每轮都重拉几十 KB 只是白等
        long nextLogRead = 0;
        int lastLogHash = 0;
        JSONObject cachedHub = null;
        int logLines = -1;
        int seenErrors = 0;

        p.stage("检查云端会话");
        // 先读日志：里面那行 QWEN21:// 是会话交出来的实情（状态/排队/地址），
        // 比单独问一次 status 更准，也少一趟请求。
        String tail0 = logTail(owner, 160000);
        lastLogHash = tail0.hashCode();
        cachedHub = parseHub(tail0);
        logLines = tail0.isEmpty() ? 0 : Math.max(1, tail0.split("\n", -1).length);
        KernelState cur = status(owner);
        if (cur.status.isEmpty() && cachedHub != null && cachedHub.has("status"))
            cur.status = cachedHub.optString("status").toUpperCase();
        p.detail("会话状态 " + (cur.status.isEmpty() ? "未知" : cur.status)
                + (logLines > 0 ? " · 日志 " + logLines + " 行" : " · 日志为空"));
        if (cur.running() || cur.queued()) {
            // 已经在跑/排着队就别再 push：push 会重启会话，等于白白再等一次冷启动
            p.stage(cur.queued() ? "会话排队中" : "会话已在运行");
        } else {
            if (cur.failed())
                p.detail("上一轮没跑完: " + (cur.failure.isEmpty() ? cur.status : cur.failure));
            // 状态不是 RUNNING 不等于没在线：status 对私有内核有时只回 COMPLETE/空。
            // 之前这里无条件 push，每次都把已经跑起来的 ComfyUI 重启一遍。
            JSONObject hub = cachedHub;
            String known = hub == null ? "" : hub.optString("url", "");
            boolean verified = !known.isEmpty() && comfyUp(known);
            p.detail(verified ? "云端还活着（" + known.replace("https://", "") + "），跳过重启"
                    : (known.isEmpty() ? "日志里没有可用地址" : "日志里有地址但服务没应"));
            if (!verified) {
                p.stage("启动云端 GPU 会话（push 内核）");
                try {
                    push(owner, apiKey);
                    p.detail("push 已送达 Kaggle，等它排 GPU");
                } catch (Api.KaggleException e) {
                    throw e;
                }
            }
        }

        String lastStage = "";
        String lastWaitNote = "";
        boolean toldPushed = false;
        long nextStatus = 0;
        KernelState lastStatus = null;
        while (System.currentTimeMillis() - t0 < timeoutMs) {
            if (p.cancelled()) throw new InterruptedException("已取消");
            long el = (System.currentTimeMillis() - t0) / 1000;

            // 日志每 5 秒读一次；status 每 30 秒才问一次——它要跨一趟 Kaggle，
            // 而日志里已经带着会话自己说的状态了。
            long now = System.currentTimeMillis();
            if (now >= nextLogRead) {
                nextLogRead = now + 5000;
                String tail = logTail(owner, 160000);
                if (tail.hashCode() != lastLogHash) {
                    lastLogHash = tail.hashCode();
                    cachedHub = parseHub(tail);
                    logLines = tail.isEmpty() ? 0 : Math.max(1, tail.split("\n", -1).length);
                }
            }
            KernelState k;
            if (now >= nextStatus || lastStatus == null) {
                nextStatus = now + 30000;
                lastStatus = status(owner);
            }
            k = lastStatus;
            if (k.status.isEmpty() && cachedHub != null && cachedHub.has("status"))
                k.status = cachedHub.optString("status").toUpperCase();
            JSONObject hub = cachedHub;
            String stage = hub == null ? "" : hub.optString("stage", "");

            // 云端自己说失败过一次就够了。以前只看状态字段，而 Kaggle 把失败会话常常标成
            // COMPLETE，于是这里一直转圈到 25 分钟超时——用户看到的「一直没反应」。
            boolean cloudError = hub != null && "error".equals(hub.optString("kind", ""));
            if (k.failed() || (cloudError && seenErrors >= 2)) {
                String why = cloudError ? (stage + ": " + hub.optString("detail", ""))
                        : (k.failure.isEmpty() ? k.status : k.failure);
                throw new Api.KaggleException("云端在「" + humanStage(stage) + "」这一步失败了: " + why
                        + " · 完整日志: kaggle.com/code/" + owner + "/" + SLUG);
            }
            if (cloudError) seenErrors++;

            if (!stage.isEmpty() && !stage.equals(lastStage)) {
                lastStage = stage;
                p.stage(humanStage(stage) + " · " + el + "s");
                if (cloudError) p.detail("云端报错: " + hub.optString("detail", ""));
            } else if (stage.isEmpty()) {
                // 空日志 + 空状态 = push 还没被 Kaggle 接住；状态有值 = 已经在跑但还没吐进度
                String q = hub == null ? "" : hub.optString("queue", "");
                String note = "等云端开口 · " + el + "s"
                        + (k.status.isEmpty() ? " · 状态未知" : " · " + k.status)
                        + (q.isEmpty() ? "" : " · " + q)
                        + (logLines < 0 ? "" : logLines == 0 ? " · 日志为空" : " · 日志 " + logLines + " 行");
                if (!note.equals(lastWaitNote)) {
                    lastWaitNote = note;
                    p.stage(note);
                } else {
                    p.detail(note);
                }
                // 说清楚「请求发出去了没有」——这正是唯一不可见、也最容易怀疑的地方
                if (!toldPushed) {
                    toldPushed = true;
                    p.detail(pushed() ? "push 已发出，Kaggle 还没开跑" : "本轮没有 push：日志无地址但会话状态是 "
                            + (k.status.isEmpty() ? "未知" : k.status));
                }
            } else {
                p.detail(humanStage(stage) + " · " + el + "s");
            }

            String url = hub == null ? "" : hub.optString("url", "");
            String token = hub == null ? "" : hub.optString("token", "");
            if (!url.isEmpty()) {
                p.stage("公网地址就绪，探活中");
                try {
                    // 日志里那条 ready 带的令牌可能是上一会话的旧值，交给 handshake 校准
                    String good = handshake(url, token);
                    Map<String, String> out = new LinkedHashMap<>();
                    out.put("url", url);
                    out.put("token", good);
                    out.put("owner", owner);
                    out.put("gpu", hub.optString("gpu", ""));
                    out.put("egress", hub.optString("egress", ""));
                    out.put("rev", String.valueOf(hub.optInt("rev", 0)));
                    if (!good.equals(token)) p.detail("令牌已按云端现时的校准");
                    current = null;
                    return out;
                } catch (Api.AuthError ae) {
                    // 隧道层面 401（cloudflared 白名单 / 网关）：不是 Key 的问题
                    p.detail("云端网关回 " + ae.getMessage() + "，继续等新地址");
                } catch (Exception e) {
                    p.detail("探活未通: " + e.getMessage());
                }
            }
            Thread.sleep(6000);
        }
        current = null;
        throw new Api.KaggleException("超时 " + (timeoutMs / 1000) + "s：云端没吐出地址 · "
                + (logLines <= 0 ? "内核日志读不到（push 没被 Kaggle 接住，或这一版日志先断在装环境）"
                : "日志共 " + logLines + " 行但没有 QWEN21:// 的地址行，多半卡在装环境或起 ComfyUI")
                + " · 网页看着一步: kaggle.com/code/" + owner + "/" + SLUG);
    }

    /**
     * 云端 ComfyUI 是否真的在应答。只探 /qwen/health 的存活位：
     * 隧道对陌生出口 IP 会回 401，那是网关挡的，不能据此说"服务没起"。
     */
    static boolean comfyUp(String base) {
        try {
            HttpURLConnection c = (HttpURLConnection) new URL(base + "/qwen/health").openConnection();
            c.setConnectTimeout(8000);
            c.setReadTimeout(8000);
            c.setRequestProperty("User-Agent", "Qwen21APK/2.0");
            int code = c.getResponseCode();
            c.disconnect();
            return code >= 200 && code < 400;
        } catch (Exception e) {
            return false;
        }
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
            case "probe-image": return "真跑一张确认能出图";
            case "handoff": return "会话常驻中";
            case "restart-comfy": return "看门狗重启服务";
            default: return s == null || s.isEmpty() ? "云端准备中" : s;
        }
    }

}
