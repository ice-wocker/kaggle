package com.cb.qwen21;

import org.json.JSONArray;
import org.json.JSONObject;

/** Qwen-Image-2.1 的极简 t2i 工作流（对应 ComfyUI 官方 t2i 子图展开后的节点） */
final class Workflow {

    private Workflow() {}

    static JSONObject build(String prompt, String negative, int width, int height,
                            int steps, long seed, double cfg, String unet, String clip, String vae,
                            String tagPrefix) throws Exception {
        return build(prompt, negative, width, height, steps, seed, cfg, 1, "euler", "simple",
                unet, clip, vae, tagPrefix);
    }

    static JSONObject build(String prompt, String negative, int width, int height,
                            int steps, long seed, double cfg, int batch, String sampler, String scheduler,
                            String unet, String clip, String vae, String tagPrefix) throws Exception {
        JSONObject wf = new JSONObject();
        wf.put("1", node("UNETLoader", "unet_name", unet, "weight_dtype", "default"));
        wf.put("2", node("CLIPLoader", "clip_name", clip, "type", "qwen_image", "device", "default"));
        wf.put("3", node("VAELoader", "vae_name", vae));

        JSONObject enc = new JSONObject();
        enc.put("class_type", "TextEncodeQwenImage21");
        JSONObject ei = new JSONObject();
        ei.put("clip", link("2", 0));
        ei.put("prompt", prompt);
        ei.put("negative_prompt", negative == null ? "" : negative);
        ei.put("resolution", 1024);
        enc.put("inputs", ei);
        wf.put("4", enc);

        JSONObject lat = new JSONObject();
        lat.put("class_type", "EmptyLatentImage");
        JSONObject li = new JSONObject();
        li.put("width", width);
        li.put("height", height);
        li.put("batch_size", Math.max(1, Math.min(4, batch)));
        lat.put("inputs", li);
        wf.put("5", lat);

        JSONObject ks = new JSONObject();
        ks.put("class_type", "KSampler");
        JSONObject ki = new JSONObject();
        ki.put("model", link("1", 0));
        ki.put("positive", link("4", 0));
        ki.put("negative", link("4", 1));
        ki.put("latent_image", link("5", 0));
        ki.put("seed", seed);
        ki.put("steps", steps);
        ki.put("cfg", cfg);
        ki.put("sampler_name", sampler == null || sampler.isEmpty() ? "euler" : sampler);
        ki.put("scheduler", scheduler == null || scheduler.isEmpty() ? "simple" : scheduler);
        ki.put("denoise", 1.0);
        ks.put("inputs", ki);
        wf.put("6", ks);

        JSONObject dec = new JSONObject();
        dec.put("class_type", "VAEDecode");
        JSONObject di = new JSONObject();
        di.put("samples", link("6", 0));
        di.put("vae", link("3", 0));
        dec.put("inputs", di);
        wf.put("7", dec);

        JSONObject save = new JSONObject();
        save.put("class_type", "SaveImage");
        JSONObject si = new JSONObject();
        si.put("images", link("7", 0));
        si.put("filename_prefix", tagPrefix);
        save.put("inputs", si);
        wf.put("8", save);
        return wf;
    }

    private static JSONObject node(String type, Object... kv) throws Exception {
        JSONObject n = new JSONObject();
        n.put("class_type", type);
        JSONObject in = new JSONObject();
        for (int i = 0; i < kv.length; i += 2) in.put((String) kv[i], kv[i + 1]);
        n.put("inputs", in);
        return n;
    }

    private static JSONArray link(String id, int slot) {
        JSONArray a = new JSONArray();
        a.put(id);
        a.put(slot);
        return a;
    }
}
