package com.cb.qwen21;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

/** ComfyUI HTTP API 的最小客户端：/system_stats /prompt /history /view /upload/image */
final class Api {
    final String base; // 例如 https://xxx.trycloudflare.com
    final String token;

    Api(String base, String token) {
        String b = base == null ? "" : base.trim();
        while (b.endsWith("/")) b = b.substring(0, b.length() - 1);
        this.base = b;
        this.token = token;
    }

    private HttpURLConnection open(String path, String method) throws Exception {
        URL u = new URL(base + path);
        HttpURLConnection c = (HttpURLConnection) u.openConnection();
        c.setRequestMethod(method);
        c.setConnectTimeout(20000);
        c.setReadTimeout(180000);
        c.setRequestProperty("User-Agent", "Qwen21APK/1.0");
        if (token != null && !token.isEmpty()) c.setRequestProperty("X-QWEN-TOKEN", token);
        return c;
    }

    static String readAll(InputStream in) throws Exception {
        ByteArrayOutputStream bo = new ByteArrayOutputStream();
        byte[] buf = new byte[16384];
        int n;
        while ((n = in.read(buf)) > 0) bo.write(buf, 0, n);
        return new String(bo.toByteArray(), StandardCharsets.UTF_8);
    }

    static byte[] readBytes(InputStream in) throws Exception {
        ByteArrayOutputStream bo = new ByteArrayOutputStream();
        byte[] buf = new byte[65536];
        int n;
        while ((n = in.read(buf)) > 0) bo.write(buf, 0, n);
        return bo.toByteArray();
    }

    String get(String path) throws Exception {
        HttpURLConnection c = open(path, "GET");
        int code = c.getResponseCode();
        InputStream in = code >= 400 ? c.getErrorStream() : c.getInputStream();
        String body = in == null ? "" : readAll(in);
        c.disconnect();
        if (code == 401 || code == 403) throw new AuthError("令牌不对");
        if (code >= 400) throw new ApiException(code, body);
        return body;
    }

    String postJson(String path, JSONObject body) throws Exception {
        HttpURLConnection c = open(path, "POST");
        c.setDoOutput(true);
        c.setRequestProperty("Content-Type", "application/json");
        c.getOutputStream().write(body.toString().getBytes(StandardCharsets.UTF_8));
        int code = c.getResponseCode();
        InputStream in = code >= 400 ? c.getErrorStream() : c.getInputStream();
        String s = in == null ? "" : readAll(in);
        c.disconnect();
        if (code == 401 || code == 403) throw new AuthError("令牌不对");
        if (code >= 400) throw new ApiException(code, s);
        return s;
    }

    byte[] view(String filename, String subfolder, String type) throws Exception {
        String q = "/view?filename=" + enc(filename) + "&subfolder=" + enc(subfolder) + "&type=" + enc(type);
        HttpURLConnection c = open(q, "GET");
        int code = c.getResponseCode();
        if (code >= 400) throw new ApiException(code, "view failed");
        byte[] b = readBytes(c.getInputStream());
        c.disconnect();
        return b;
    }

    static String enc(String s) {
        try {
            return java.net.URLEncoder.encode(s == null ? "" : s, "UTF-8");
        } catch (Exception e) {
            return "";
        }
    }

    String stats() throws Exception {
        return get("/system_stats");
    }

    /** 提交工作流，返回 prompt_id */
    String submit(JSONObject workflow) throws Exception {
        JSONObject req = new JSONObject();
        req.put("prompt", workflow);
        req.put("client_id", "apk-" + UUID.randomUUID().toString().substring(0, 8));
        JSONObject r = new JSONObject(postJson("/prompt", req));
        if (r.has("node_errors") && r.getJSONObject("node_errors").length() > 0)
            throw new IllegalStateException("节点参数错误: " + r.getJSONObject("node_errors").toString());
        return r.getString("prompt_id");
    }

    /** 轮询直到完成，返回 /history 里的那一份 */
    JSONObject wait(String promptId, long timeoutMs, Progress p) throws Exception {
        long t0 = System.currentTimeMillis();
        while (System.currentTimeMillis() - t0 < timeoutMs) {
            if (p != null && p.cancelled()) throw new InterruptedException("已取消");
            String body = get("/history/" + promptId);
            JSONObject h = new JSONObject(body);
            if (h.has(promptId)) {
                JSONObject e = h.getJSONObject(promptId);
                JSONObject st = e.optJSONObject("status");
                String s = st == null ? "" : st.optString("status_str");
                if ("error".equals(s)) throw new IllegalStateException("生成失败: " + st.optJSONArray("messages"));
                if (st != null && st.optBoolean("completed")) return e;
            }
            if (p != null) p.tick((System.currentTimeMillis() - t0) / 1000);
            Thread.sleep(1500);
        }
        throw new IllegalStateException("超时未完成");
    }

    static JSONArray images(JSONObject historyEntry) {
        JSONArray out = new JSONArray();
        JSONObject o = historyEntry.optJSONObject("outputs");
        if (o == null) return out;
        java.util.Iterator<String> it = o.keys();
        while (it.hasNext()) {
            JSONObject n = o.optJSONObject(it.next());
            if (n == null) continue;
            JSONArray arr = n.optJSONArray("images");
            if (arr == null) continue;
            for (int i = 0; i < arr.length(); i++) out.put(arr.optJSONObject(i));
        }
        return out;
    }

    /**
     * 提交加密任务。整个信封（提示词 + 参数）在端上封好，明文不出设备，
     * 云端只负责解封和执行。返回任务 id。
     */
    String submitSealed(byte[] envelope) throws Exception {
        JSONObject req = new JSONObject();
        req.put("t", android.util.Base64.encodeToString(envelope, android.util.Base64.NO_WRAP));
        // 信封里没有任何凭据字段，鉴权只能走请求头
        req.put("k", token == null ? "" : token);
        String body = postJson("/qwen/generate", req);
        JSONObject o = new JSONObject(body);
        if (o.has("error")) throw new IllegalStateException(o.getString("error"));
        return o.getString("id");
    }

    JSONObject task(String id) throws Exception {
        return new JSONObject(get("/qwen/task/" + enc(id)));
    }

    interface Progress {
        void tick(long sec) throws Exception;
        boolean cancelled();
    }

    /** 凭据不对：需要重新拿 Key / 令牌，重试没有意义。 */
    static final class AuthError extends Exception {
        AuthError(String m) { super(m); }
    }

    /** Kaggle 侧的业务失败（push 被拒、内核报错、等待超时）。 */
    static final class KaggleException extends Exception {
        KaggleException(String m) { super(m); }
    }

    static final class ApiException extends Exception {
        final int code;
        ApiException(int code, String body) {
            super("HTTP " + code + ": " + (body == null ? "" : body.substring(0, Math.min(200, body.length()))));
            this.code = code;
        }
    }
}
