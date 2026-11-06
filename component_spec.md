# Component Specification & Validation Predicates
**Date:** 6 Nov 2026  

| # | Component | Interface / API | CMake / Flag | Mandatory MLIR Passes | Validation Predicate |
|---|-----------|----------------|--------------|------------------------|----------------------|
| **C1** | GGUF → MLIR front‑end | `iree‑import‑gguf <file> -o <mlir>` or custom parser | `-DGGUF_PATH="…"` | – | `has_valid_quant_token(tok)` |
| **C2** | MLIR optimisation pipeline | `mlir-opt` pass sequence | `‑mlir‑tile=128`, `‑mlir‑fuse‑attention`, `‑mlir‑add‑bmc`, `‑mlir‑add‑mtp`, `‑mlir‑add‑lmcache`, `‑mlir‑add‑moe‑router`, `‑mlir‑add‑expert‑cache`, `‑mlir‑add‑prefetch`, `‑vectorize‑simd‑width=256`, `‑mlir‑add‑selective‑dequant` | `all_passes_passed(mlir_module)` |
| **C3** | LLVM‑IR legalisation | `opt -march=znver3 -mno-avx512 -mno-vnn -vectorize‑simd‑width=256` | auto | – | `no_avx512_or_vnnil_in_ir(ir)` |
| **C4** | Fused quantised kernel `q4_matmul.c` | `void q4_matmul(const float* scales, const int32_t* A, const float* B_scales, const int32_t* B, float* out, int M,int N,int K)` | `-DQUANTIZED_GEMV`; `‑DTERNARY`, `‑DINT8`, `‑DGPTQ`, `‑DAWQ`, `‑DBF16` (float path) as needed | – | `kernel_supports_quant(quant_type)` |
| **C5** | ORC‑lite host driver `main.cpp` | Loads `model.o`, resolves `q4_matmul`, `draft_step`, `verify_step`, `init_expert_cache`, `load_expert`, `release_expert`, `selective_dequant_start/_tile/end` | `-DMEO` (MoE), `-DBMC`, `-DLM_CACHE`, `-DSPEC‑WIDTH=N`, `-DSELECTIVE_DEQUANT`, `-DGPTQ`, `-DAWQ`, `-DBF16` (float path) | – | `all_symbols_resolved(driver)` |
| **C5** | Selective de‑quant handler | Functions `selective_dequant_start()`, `selective_dequant_tile(layer, tile)`, `selective_dequant_end()` | `-DSELECTIVE_DEQUANT` | – | `selective_dequant_valid()` |
| **C6** | KV‑cache (BMC) | `init_kv_cache(size, mode)`, `kv_read/write` | `-DBMC` | `‑mlir‑add‑kv‑cache` | `bmc_valid_for_quant(quant_type)` |
| **C6** | LM‑cache + Mixed‑precision | `init_lm_cache()`, `lmcache_fetch(layer, tile)` | `-DLM_CACHE` | `‑mlir‑add‑lmcache` | `lmcache_quant_valid(quant_type)` |
| **C6** | Mixture‑of‑Experts | `init_expert_cache(ram_count)`, `load_expert(id)`, `release_expert(id)` | `-DMEO`, `-DEXPERT_RAMLIMIT=120` | `‑mlir‑add‑moe‑router`, `‑mlir‑add‑expert‑cache` | `moe_expert_valid(id, quant_type)` |
| **C7** | Sherlock‑light Tracing | Compile‑time `‑DSHERLOCK_LITE` | – | – | `trace_initialised()` |
| **C8** | Breadcrumb Debug Tags | `__BREADCRUMB("component", "phase")` macro | `-DDEBUG_BREADCRUMBS` | – | `breadcrumbs_enabled()` |
| **C9** | Selective de‑quant metadata | Per‑tile flag `used_in_current_batch`; bitmap stored in GGUF extension; read at init. | `-DSELECTIVE_DEQUANT` | `‑mlir‑add‑selective‑dequant` (new pass) | `has_selective_dequant_metadata(gguf)` |
| **C10** | GPTQ/AWQ kernel lowering | MLIR pass `‑mlir‑add‑gptq‑lowering` / `‑mlir‑add‑awq‑lowering` that emits `gptq_ matmul` / `awq_matmul` ops with per‑group/per‑token scale metadata. | `-DGPTQ` / `-DAWQ` | `gptq_quant_valid()` / `awq_quant_valid()` | Checks that scale metadata exists and group/token masks are within bounds. |

---