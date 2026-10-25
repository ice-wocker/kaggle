package com.cb.qwen21;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/**
 * 与 ComfyUI custom_node comfy_node_crypt 完全一致的编解码。
 * 加解密都在端上算，网络上看不到明文。
 */
final class Crypto {
    static final int MAGIC = 0xCB01;

    private Crypto() {}

    static byte[] sha256(byte[] in) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(in);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    static byte[] key(String seed) {
        return sha256(seed.getBytes(StandardCharsets.UTF_8));
    }

    private static byte[] keystream(byte[] key, String tag, int n) {
        byte[] out = new byte[n];
        int off = 0, ctr = 0;
        byte[] t = tag.getBytes(StandardCharsets.US_ASCII);
        while (off < n) {
            byte[] buf = new byte[key.length + t.length + 4];
            System.arraycopy(key, 0, buf, 0, key.length);
            System.arraycopy(t, 0, buf, key.length, t.length);
            // 小端，与 Python struct.pack("<I") 对齐
            buf[buf.length - 4] = (byte) ctr;
            buf[buf.length - 3] = (byte) (ctr >>> 8);
            buf[buf.length - 2] = (byte) (ctr >>> 16);
            buf[buf.length - 1] = (byte) (ctr >>> 24);
            byte[] d = sha256(buf);
            int c = Math.min(d.length, n - off);
            System.arraycopy(d, 0, out, off, c);
            off += c;
            ctr++;
        }
        return out;
    }

    private static int[] subkeys(byte[] key, int rounds) {
        int[] sk = new int[rounds];
        for (int i = 0; i < rounds; i++) {
            byte[] buf = new byte[key.length + 2];
            System.arraycopy(key, 0, buf, 0, key.length);
            buf[key.length] = (byte) 'k';
            buf[key.length + 1] = (byte) ('0' + i);
            byte[] d = sha256(buf);
            sk[i] = ((d[0] & 0xFF) << 24) | ((d[1] & 0xFF) << 16) | ((d[2] & 0xFF) << 8) | (d[3] & 0xFF);
        }
        return sk;
    }

    private static byte[] xor(byte[] a, byte[] b) {
        byte[] r = new byte[a.length];
        for (int i = 0; i < a.length; i++) r[i] = (byte) (a[i] ^ b[i]);
        return r;
    }

    /** 4 轮 ARX Feistel，8 字节分组 PKCS#7 */
    static byte[] encrypt(byte[] plain, byte[] key) {
        int pad = 8 - (plain.length % 8);
        byte[] data = new byte[plain.length + pad];
        System.arraycopy(plain, 0, data, 0, plain.length);
        for (int i = plain.length; i < data.length; i++) data[i] = (byte) pad;
        int[] sk = subkeys(key, 4);
        byte[] out = new byte[data.length];
        for (int off = 0; off < data.length; off += 8) {
            int l = be32(data, off), r = be32(data, off + 4);
            for (int k : sk) {
                int nl = r;
                r = l ^ ((Integer.rotateLeft(r, 3)) + k);
                l = nl;
            }
            put32(out, off, l);
            put32(out, off + 4, r);
        }
        return xor(out, keystream(key, "diffuse", out.length));
    }

    static byte[] decrypt(byte[] cipher, byte[] key) {
        int[] sk = subkeys(key, 4);
        byte[] data = xor(cipher, keystream(key, "diffuse", cipher.length));
        byte[] out = new byte[data.length];
        for (int off = 0; off < data.length; off += 8) {
            int l = be32(data, off), r = be32(data, off + 4);
            for (int i = sk.length - 1; i >= 0; i--) {
                int nr = l;
                l = r ^ (Integer.rotateLeft(l, 3) + sk[i]);
                r = nr;
            }
            put32(out, off, l);
            put32(out, off + 4, r);
        }
        int pad = out[out.length - 1] & 0xFF;
        if (pad < 1 || pad > 8) throw new IllegalStateException("bad padding");
        byte[] r2 = new byte[out.length - pad];
        System.arraycopy(out, 0, r2, 0, r2.length);
        return r2;
    }

    private static int be32(byte[] b, int o) {
        return ((b[o] & 0xFF) << 24) | ((b[o + 1] & 0xFF) << 16) | ((b[o + 2] & 0xFF) << 8) | (b[o + 3] & 0xFF);
    }

    private static void put32(byte[] b, int o, int v) {
        b[o] = (byte) (v >>> 24);
        b[o + 1] = (byte) (v >>> 16);
        b[o + 2] = (byte) (v >>> 8);
        b[o + 3] = (byte) v;
    }

    /** 任务 -> 32 字节码字（magic 4 + len 2 + 密文） */
    static byte[] encodeTask(String prompt, byte[] key) {
        byte[] p = prompt.getBytes(StandardCharsets.UTF_8);
        byte[] ct = encrypt(p, key);
        byte[] blob = new byte[6 + ct.length];
        blob[0] = (byte) 0xCB;
        blob[1] = 0x21;
        blob[2] = 0x01;
        blob[3] = 0x00;
        blob[4] = (byte) (ct.length >>> 8);
        blob[5] = (byte) ct.length;
        System.arraycopy(ct, 0, blob, 6, ct.length);
        return blob;
    }

    static String decodeTask(byte[] blob, byte[] key) {
        if (blob.length < 7 || (blob[0] & 0xFF) != 0xCB || (blob[1] & 0xFF) != 0x21)
            throw new IllegalStateException("格式不匹配");
        int n = ((blob[4] & 0xFF) << 8) | (blob[5] & 0xFF);
        byte[] ct = new byte[n];
        System.arraycopy(blob, 6, ct, 0, n);
        return new String(decrypt(ct, key), StandardCharsets.UTF_8);
    }

    static String hex(byte[] b) {
        StringBuilder sb = new StringBuilder();
        for (byte x : b) sb.append(String.format("%02x", x));
        return sb.toString();
    }

    // ---------- 密文图：位反转铺满的码字画布 ----------
    /** 密文图最小边长；码字多时翻倍，边长始终是 2 的幂。 */
    static final int MIN_SIDE = 32;
    /** 填充码字：高 4 位标记位，不与真实数据混淆。 */
    static final int FILL = 0xE000;

    /** 码字数 -> 画布边长。边长取 2 的幂，位反转才是严格双射。 */
    static int sideFor(int count) {
        int side = MIN_SIDE;
        while (side * side < count) side *= 2;
        return side;
    }

    /**
     * 码字序号 -> 像素线性下标。对 side*side 个格子做位反转（Van der Corput），
     * 任意前缀都铺得最均匀。与 ComfyUI 端 _slot() 完全一致。
     */
    static int slot(int i, int side) {
        int n = side * side;
        int bits = 32 - Integer.numberOfLeadingZeros(n) - 1; // n 是 2 的幂
        int x = i % n;
        int r = 0;
        for (int b = 0; b < bits; b++) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        return r;
    }

    /** 把码字写进 ARGB 像素：G=高字节, B=低字节, R=两者异或。 */
    static int[] renderWords(int[] words) {
        int side = sideFor(words.length);
        int n = side * side;
        int[] px = new int[n];
        java.util.Arrays.fill(px, encodeWord(FILL));
        for (int i = 0; i < words.length && i < n; i++) {
            px[slot(i, side)] = encodeWord(words[i]);
        }
        return px;
    }

    static int encodeWord(int w) {
        int g = (w >> 8) & 0xFF, b = w & 0xFF;
        return 0xFF000000 | ((g ^ b) << 16) | (g << 8) | b;
    }

    /** 从 ARGB 像素还原前 count 个码字。 */
    static int[] unrenderWords(int[] argb, int count) {
        int side = (int) Math.round(Math.sqrt(argb.length));
        int[] out = new int[count];
        for (int i = 0; i < count; i++) {
            int c = argb[slot(i, side)];
            int g = (c >> 8) & 0xFF;
            int b = c & 0xFF;
            out[i] = (g << 8) | b;
        }
        return out;
    }

    /** 任务明文 -> 码字数组（含 6 字节头）。 */
    static int[] encodeTaskWords(String prompt, byte[] key) {
        byte[] blob = encodeTask(prompt, key);
        int[] words = new int[blob.length];
        for (int i = 0; i < blob.length; i++) {
            words[i] = FILL | ((blob[i] & 0xFF) << 4);
        }
        return words;
    }

    /** 码字数组 -> 任务明文。 */
    static String decodeTaskWords(int[] words, byte[] key) {
        byte[] blob = new byte[words.length];
        for (int i = 0; i < words.length; i++) {
            blob[i] = (byte) ((words[i] >> 4) & 0xFF);
        }
        return decodeTask(blob, key);
    }
}
