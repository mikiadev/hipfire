---
title: GSQ-RCO IQ3_S experiment branch: why exp/gsq-rco-iq3s exists
date: 2026-09-08
tags: [Created 2026-09-08 for IQ3_S loader support work, separate from feat/escha-w2 Escha codec work. Context: Kairic IU4 lane cannot serve Escha trellis codes (integer MMA needs uniform-affine grids); IQ3_S uniform-affine grid IS integer-MMA compatible. Plan: (1) extend GgmlType/tensor_to_f32 with IQ3_S dequant from llama.cpp-escha, (2) evaluate dequant-then-MQ recipe, (3) Astrea hardware-aware policy, (4) IU4 research arm later. Blocker: no IQ3_S GGUF on local disk, slow link, no BF16 base. Tags: gsqrco, iq3s, iu4, branch]
---

<write the finding here; terse; link related notes by slug>
