---
title: GSQ-RCO IQ3_S loader complete, first .hfq written
date: 2026-09-08
tags: [gsqrco, iq3s, loader, hfq, dequant]
---

Loader unlocks the released file: 7 decoders ported (Q2_K, IQ2_XXS/XS/S, IQ3_XXS, IQ4_XS, IQ1_M) + IQ3_S verified 512/512 vs C header. Hybrid naming (attn_qkv, ssm_*, ssm_a/ssm_dt.bias spellings) + qwen35 GGUF config (16/48 heads, layer_types from tensor table). First conversion: /tmp/gsqrco-iq3s.hfq (851 tensors, arch 5, 14GB MQ4, 305 F16 finite). Known: 64 post_attention_layernorm names need rerun (fix committed after). Next: rerun conversion, serve smoke, KLD-gate vs F16, Astrea policy. Related: gsqrco-branch-intent.
