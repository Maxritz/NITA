# Internal Re‑check: Missing & Invalid Component Detection
**Date:** 6 Nov 2026  

## 1️⃣ Compile‑time static asserts

```cpp
static_assert(
    QUANT_TYPE == Q4_K_M || QUANT_TYPE == Q5_K_M || QUANT_TYPE == Q3_K_M ||
    QUANT_TYPE == Q2_K  || QUANT_TYPE == Q1_K  || QUANT_TYPE == Q8_0 ||
    QUANT_TYPE == Ternary,
    "Unsupported quant type: will fall back to de‑quant path (not allowed)");
```

## 2️⃣ Runtime component checklist

```cpp
bool validate_engine_components() {
    bool ok = true;
    if (!has_valid_gguf_token())                     { ok = false; log_error("missing/invalid GGUF token"); }
    if (!mlir_passes_ok())                           { ok = false; log_error("MLIR pass failure"); }
    if (!kernel_supports_quant(QUANT_TYPE))          { ok = false; log_error("kernel lacks quant support"); }
    if (!trace_initialised())                        { ok = false; log_error("sherlock‑light trace init failed"); }
    if (!breadcrumbs_enabled() && DEBUG_BREADCRUMBS) { ok = false; log_error("breadcrumbs forced but disabled"); }
    // MoE specific
    if (ENABLE_MOE && !moe_expert_valid_all())       { ok = false; log_error("MoE expert validation failed"); }
    // Selective de‑quant specific
    if (SELECTIVE_DEQUANT && !selective_dequant_valid()) { ok = false; log_error("selective de‑quant metadata missing or invalid"); }
    // GPTQ/AWQ specific
    if (ENABLE_GPTQ && !gptq_quant_valid())          { ok = false; log_error("GPTQ quant validation failed"); }
    if (ENABLE_AWQ && !awq_quant_valid())            { ok = false; log_error("AWQ quant validation failed"); }
    // BF16/FP16 float path specific
    if (ENABLE_BF16 && !has_float_matmul_kernel())    { ok = false; log_error("BF16 float matmul kernel not included"); }
    if (ENABLE_FP16 && !has_float_matmul_kernel())    { ok = false; log_error("FP16 float matmul kernel not included"); }
    return ok;
}
// Called at start‑up in main():
assert(validate_engine_components() && "Engine component validation failed – aborting");
```

## 3️⃣ Enforcement rules (quoted from the sherlock‑it skill, adapted)

- Never assume a kernel is optimal because it uses a specialised instruction.
- Never equate aggregate device time with individual invocation latency.
- Never confuse theoretical peak performance with achievable workload performance.
- Never accept an optimisation based on a single noisy measurement.
- Never claim a root cause without evidence.
- Never stop at the first bottleneck if another significant bottleneck remains.
- Never discard the original baseline or correctness tests.
- Always redirect the investigation towards the highest‑impact measurable performance gap.

---