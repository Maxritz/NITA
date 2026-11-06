# NITA (NC Inference Tool for All)

## Overview
Pure LLVM‑IR CPU‑only inference engine for GGUF models, AVX2‑only (AMD Zen3), no runtime de‑quant of full f32 weight matrix.

Supports **all** quantisation types via CMake flags:
- Q4_K_M (default)
- TERNARY {-1,0,1}
- GPTQ (4‑bit grouped)
- AWQ (activation‑aware)
- BINARY_NET (BitNet)
- BF16 / FP16 (float path, scalar)

## Build Requirements
- GCC / MinGW 13+ (tested with Strawberry Perl GCC)
- CMake 3.20+
- Optional: LLVM (not required for current pure‑C implementation)

## Quick Start
```bash
# 1. Configure
mkdir -p build && cd build
cmake .. \
  -DQUANTIZED_GEMV=ON \
  -DQUANT_TYPE=Q4_K_M   # or TERNARY, GPTQ, AWQ, BINARY_NET, BF16, FP16

# 2. Build
cmake --build . -j$(nproc)

# 3. Run
./minimal_engine.exe
```

## CMake Options
| Flag | Effect |
|------|--------|
| `-DQUANTIZED_GEMV` | Required base flag |
| `-DQUANT_TYPE=Q4_K_M` | Default 4‑bit packed kernel |
| `-DQUANT_TYPE=TERNARY` | Ternary weights {-1,0,1} |
| `-DQUANT_TYPE=GPTQ` | 4‑bit grouped GPTQ |
| `-DQUANT_TYPE=AWQ` | Activation‑aware weight quantisation |
| `-DQUANT_TYPE=BINARY_NET` | BitNet 1‑bit signs |
| `-DQUANT_TYPE=BF16` / `-DQUANT_TYPE=FP16` | Float matmul (scalar) |
| `-DMXFP8` | Forces compile‑time error (not on AVX2) |

## Output
The binary prints a single line on stdout:
```
[minimal_engine] Kernel executed, out=<value>
```
The value depends on the selected quant type and the tiny test matrices compiled in.

## License
MIT‑style – feel free to use, modify, and extend.

## Contact / Project
Project name: **NITA** (NC Inference Tool for All)