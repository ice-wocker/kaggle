package com.cb.qwen21;

/** 提示词预设与自动改写：把「一句大白话」补成模型听得懂的结构化描述。 */
final class Presets {

    private Presets() {}

    static final String[] STYLE_NAMES = {
            "自动", "写实摄影", "电影感", "日系插画", "赛博朋克", "胶片质感", "3D 渲染", "线稿上色"
    };

    /** 追加在正向词后面的风格词（空串 = 不改写）。 */
    static String style(String name) {
        switch (name == null ? "" : name) {
            case "写实摄影": return "raw photo, natural skin texture, fine detail, 85mm, f/1.8, soft window light, high dynamic range";
            case "电影感": return "cinematic still, anamorphic lens, dramatic rim light, shallow depth of field, muted color grade, 35mm film grain";
            case "日系插画": return "japanese illustration, clean line art, soft cel shading, pastel palette, delicate light bloom";
            case "赛博朋克": return "cyberpunk, neon reflections, wet asphalt, volumetric fog, teal and magenta grade, dense city lights";
            case "胶片质感": return "analog film, kodak portra, visible grain, slight halation, warm highlights, natural contrast";
            case "3D 渲染": return "stylized 3d render, subsurface scattering, soft studio setup, octane look, crisp micro detail";
            case "线稿上色": return "flat color illustration, bold outlines, limited palette, clean shading";
            default: return "";
        }
    }

    /**
     * 是否该自动加质量词。用户已经写得很具体时不再堆料，
     * 否则会把他的意图稀释掉。
     */
    static boolean needsQualityWords(String prompt) {
        if (prompt == null) return false;
        String p = prompt.toLowerCase();
        if (p.length() > 260) return false;
        String[] owned = {"photo", "photograph", "illustration", "render", "shot on", "8k", "mp", "lens",
                "摄影", "特写", "景深", "插画", "渲染", "构图"};
        for (String w : owned) if (p.contains(w)) return false;
        return true;
    }

    static String qualityWords() {
        return "highly detailed, sharp focus, professional composition, balanced lighting, rich texture";
    }

    /** NSFW 场景下更实用的默认负面词：只压解剖与画质崩坏，不压风格。 */
    static String nsfwNegative() {
        return "lowres, blurry, jpeg artifacts, watermark, signature, text, extra fingers, fused fingers, "
                + "extra limbs, malformed hands, deformed face, asymmetric eyes, bad anatomy";
    }

    /** 允许 NSFW 时不该残留的自我审查词。 */
    static final String[] CENSOR_WORDS = {
            "nsfw", "nude", "naked", "explicit", "sexual", "adult content", "porn",
            "裸露", "色情", "限制级", "18+"
    };

    /** 从负面词里剥掉 NSFW 屏蔽词（逗号分段，命中即删）。 */
    static String stripCensor(String negative) {
        if (negative == null || negative.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        for (String raw : negative.split("[,，]")) {
            String item = raw.trim();
            if (item.isEmpty()) continue;
            String low = item.toLowerCase();
            boolean drop = false;
            for (String c : CENSOR_WORDS) {
                if (low.contains(c)) { drop = true; break; }
            }
            if (drop) continue;
            if (sb.length() > 0) sb.append(", ");
            sb.append(item);
        }
        return sb.toString();
    }

    static String compose(String prompt, String styleName, boolean autoQuality) {
        String p = prompt == null ? "" : prompt.trim();
        if (p.isEmpty()) return "";
        StringBuilder sb = new StringBuilder(p);
        String s = style(styleName);
        if (!s.isEmpty()) sb.append(", ").append(s);
        if (autoQuality && needsQualityWords(p)) sb.append(", ").append(qualityWords());
        return sb.toString();
    }
}
