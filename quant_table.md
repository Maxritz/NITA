# Quantisation Mandate Table (All Quants q1‑q8, ternary, bitnet, bf16, awq, etc.)
**Date:** 6 Nov 2026  
**Target:** AMD Zen 3 (AVX2‑only, ‑mno‑avx512 ‑mno‑vnn), **selective de‑quant allowed** (only used portions de‑quantised, rest stay packed), cross‑platform (Linux / Windows)

| # | Quant Type | Bit‑width / Layout | GGUF Token / Tag | Per‑Tile Scale? | **Supports Selective De‑quant?** | AVX2‑only Kernel Note | CMake Toggle |
|---|------------|--------------------|------------------|-----------------|--------------------------------|------------------------|--------------|
| **1** | **Q1_K**   | 1.58 bit (packed signed) | `q1_k` | Yes | **Yes** – only used tiles de‑quantised; rest stay packed. | Custom packed‑int dot‑product in `q4_matmul.c` | `‑DQUANTIZED_GEMV` |
| **2** | **Q2_K**   | 2.35 bit (packed) | `q2_k` | Yes | **Yes** – same tile‑level approach. | Same kernel, larger accumulator | `‑DQUANTIZED_GEMV` |
| **3** | **Q3_K_M** | 3.53 bit (packed) | `q3_k_m` | Yes | **Yes** – tile‑level de‑quant. | Same kernel, 8‑wide nibble logic | `‑DQUANTIZED_GEMV` |
| **4** | **Q4_K_M** | 4.53 bit (packed) | `q4_k_m` | Yes | **Yes** – engine default; tile‑level de‑quant. | Core kernel; all optimisations target this | `‑DQUANTIZED_GEMV` |
| **5** | **Q5_K_M** | 5.53 bit (packed) | `q5_k_m` | Yes | **Yes** – higher quality; tile‑level de‑quant. | Same kernel, wider accumulator | `‑DQUANTIZED_GEMV` |
| **6** | **Q6_K**   | 6.57 bit (packed) | `q6_k` | Yes | **Yes** – rare; tile‑level de‑quant. | Same kernel, carry logic | `‑DQUANTIZED_GEMV` |
| **7** | **Q8_0**   | 8 bit exact (no pack offset) | `q8_0` | Yes (scale = 1.0, identity) | **Yes** – pure integer loads, no MAC needed; can skip de‑quant for inactive tiles. | Tile‑reads only | `‑DQUANTIZED_GEMV` |
| **8** | **Binary Net** (BitNet v1) | 1 bit (sign packed) | `binary` | Yes (sign‑scale) | **Yes** – only used bits de‑quantised; rest stay packed. | Special‑case kernel | `‑DBINARY_NET` |
| **9** | **Ternary** (‑1, 0, +1) | 1.58 bit packed, symmetry‑forced | `ternary` / `bitnet‑v2` | Yes (values ∈ {‑1,0,1}, per‑tile scale) | **Yes** – only non‑zero tiles de‑quantised. | `#if TERNARY` branch in `q4_matmul.c` | `‑DTERNARY` |
| **10** | **GPTQ** (4‑bit grouped + selective de‑quant) | Variable grouped 4‑bit + per‑tile scales | `gptq` | Yes (per‑tile scales, de‑quant only for used groups) | **Yes** – extend kernel with `#if GPTQ` branch; scales per‑tile; groups not used stay packed. | – |
| **11** | **AWQ** (Activation‑Aware selective de‑quant) | 4‑bit + per‑token/per‑group scales | `awq` | Yes (per‑token/per‑group scales, de‑quant only for active tokens/groups) | **Yes** – extend kernel with `#if AWQ` branch; extra scale look‑up for weight + activation. | – |
| **12** | **BF16** (Brain Float16) | 16 bit (pos‑exp) | `bf16` | **No** – requires a separate float matmul kernel; if added, selective per‑layer de‑quant is possible but bloats RAM with float buffers. | **Conditional** – add `-DBF16` and a float matmul path; otherwise excluded. | – |
| **12** | **FP16** (IEEE float16) | 16 bit IEEE | `fp16` | **No** – same as BF16. | **Conditional** – same as BF16; excluded by default unless `-DBF16`/`‑DFP16` is set and a float kernel is included. | – |
| **13** | **MXFP8 / E5M2** | 8‑bit custom float | `mxfp8` | **No** – not native to Zen 3 AVX2; would require software float emulation or a dedicated float backend. | **No** – excluded from the pure‑IR integer contract. | – |

> **Validation Rule (mandatory)** – A quant type is **valid** for this engine ⇔  
> `(type ∈ {Q1_K…Q8_0, Binary, Ternary, GPTQ, AWQ}) ∧ (engine compiled with corresponding ‑D flag) ∧ (no full‑model f32 matrix materialisation; only selective tile/token‑level de‑quant)`  
> BF16/FP16/MXFP8 are **conditionally** supported only if a float matmul kernel is included and `-DBF16`/`-DFP16` is set; otherwise they are **invalid** for the pure‑IR no‑dequant contract.

---