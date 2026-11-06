# Pure‑LLVM‑IR CPU‑only GGUF Inference Engine – Quick Start Guide
**Date:** 6 Nov 2026  

## 1️⃣ Folder Layout
```
LLVM-IR/
├─ weights/            ← GGUF model files (e.g. llama-2-7B-q4_k_m.gguf)
├─ mlir/               ← intermediate MLIR files
├─ ir/                 ← lowered LLVM IR (.ll, .bc, .o)
├─ src/
│   ├─ main.cpp        ← ORC‑lite driver
│   └─ q4_matmul.c     ← fused quantised kernel
├─ build/              ← CMake output
├─ quant_table.md      ← quant type mandate table
├─ component_spec.md   ← component specifications
├─ sherlock_spec.md    ← tracing & breadcrumbs
├─ validation_spec.md  ← component re‑check
└─ CMakeLists.txt
```

## 2️⃣ CMake Configuration
```bash
mkdir -p build && cd build
cmake .. \
  -DLLVM_ROOT=/usr/lib/llvm-18 \          # adjust to your LLVM install
  -DENABLE_BMC=ON \
  -DLM_CACHE=ON \
  -DMEO=ON \          # enable MoE hot‑expert handling
  -DSPEC_WIDTH=4 \    # speculative decode width (2‑4)
  -DDEBUG_BREADCRUMBS=ON \
  -DSHERLOCK_LITE=ON
cmake --build . -j$(nproc)
```

## 3️⃣ Run the Engine
```bash
./bin/minimal_inference ir/model.o "Once upon a time" 64
```
- The binary first prints a validation assert (all components OK).
- Then it outputs the sherlock‑light table to stdout.
- Breadcrumbs appear on `stderr` if `DEBUG_BREADCRUMBS` was compiled in.

## 4️⃣ Toggle Features at Runtime (CMake flags)
| Feature | CMake Flag | Effect |
|---|---|---|
| BMC KV‑cache | `-DBMC` | Eliminates per‑token `memcpy`; redundant rows for in‑place update |
| LM‑cache | `-DLM_CACHE` | Keeps quantised weight tiles in RAM; layer‑wise mixed‑precision |
| Mixture‑of‑Experts | `-DMEO` | Keeps `E` experts in L3 (~120 MB), loads/rest from disk LRU |
| Speculative decode | `-DSPEC_WIDTH=N` | Draft‑then‑verify with N tokens (N=2‑4) |
| Hot‑cold KV buckets | `-DHC_BUCKET` | Keeps hot tiles in L3, spills cold to RAM |
| Prefetching | `-DPREFETCH` | `llvm.prefetch` before next‑token fetch |
| LTCG | `-DLTO` | Whole‑program optimisation, constant‑folding of tile scales |

## 4️⃣ Validation on Startup
The binary runs `validate_engine_components()` at launch and aborts with a clear message if any check fails (missing GGUF token, MLIR pass failure, kernel quant mismatch, trace init failure, MoE expert validation). This is the “internal re‑check” demanded by the user.

---