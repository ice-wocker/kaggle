package com.cb.qwen21;

import android.util.Base64;

import java.nio.charset.StandardCharsets;

/** 编码小工具：Basic 头与 base64url 载荷。 */
final class Auth {

    private Auth() {}

    static String b64(String s) {
        return Base64.encodeToString(s.getBytes(StandardCharsets.UTF_8), Base64.NO_WRAP);
    }

    /** base64url 无填充：与 Python 的 urlsafe_b64encode().rstrip("=") 对齐。 */
    static byte[] unb64(String s) {
        String t = s.replace('-', '+').replace('_', '/');
        while (t.length() % 4 != 0) t += "=";
        return Base64.decode(t, Base64.NO_WRAP);
    }
}
