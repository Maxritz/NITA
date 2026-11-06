# Build‑Fix Skill – NITA Project
**Purpose**: Systematic resolution of compile‑time and link‑time errors using ECC‑structured diagnostics.

## Command: `/ecc:build-fix <error-description>`
**When to use**: Build fails after adding a new quant‑type flag, changing CMake options, or modifying `q4_matmul.c`.

### Diagnostic Pattern
1. **Read the full compiler output** – do not stop at the first error line.
2. **Categorise the error**:
   - **Missing macro**: `QUANT_TYPE` not defined → re‑run `cmake .. -DQUANT_TYPE=<value>`
   - **Conflicting function definitions** → ensure only ONE `QUANT_TYPE` is active; check `CMakeLists.txt` derivation logic.
   - **Intrinsic mismatch** → verify `-mavx2` is set, `__AVX2__` is available, or fall back to scalar.
   - **Linker `WinMain` error** → add `src/main.c` with `main()` definition, or add `-mwindows` flag if GUI.
   - **`roundf` implicit declaration** → add `#include <math.h>`.

### Example Error‑Flow
```
ERROR: conflicting types for 'q4_matmul_impl_ternary'
→ Check: two -D flags enabled (TERNARY + Q4_K_M)
→ Fix: disable one flag: cmake .. -DTERNARY=OFF (keep only one QUANT_TYPE)
→ Rebuild: cmake --build .
→ Verify: ./minimal_engine.exe outputs expected value
```

### Common Fix‑Its Catalog
| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `error: #error "QUANTIZED_GEMV must be defined` | `-DQUANTIZED_GEMV` omitted from CMake | add `-DQUANTIZED_GEMV` to cmake command |
| `error: conflicting types for 'q4_matmul_kernel'` | Multiple `#if defined(...)` branches define the function | enable exactly one `QUANT_TYPE`; refactor to single dispatch (current design) |
| `warning: implicit declaration of 'roundf'` | `#include <math.h>` missing | add `#include <math.h>` at top of `q4_matmul.c` |
| `error: unrecognized command-line option '-mno-avx512'` | GCC version does not support that flag | replace with `-mno-avx512f` or simply `-mavx2` |
| `error: MXFP8 is not supported` | `-DMXFP8=ON` selected on AVX2 path | remove `-DMXFP8` or keep it – it will force compile error as designed |
| `fatal: linker: undefined reference to 'WinMain'` | C entry point not defined for MinGW | ensure `src/main.c` defines `int main(void)` |

### Post‑Fix Checklist
- [ ] Build passes with `cmake --build . -j$(nproc)` (zero errors, only permitted warnings)
- [ ] `./minimal_engine.exe` runs and produces the expected output line
- [ ] No new `-Werror` style warnings that halt build
- [ ] Commit with `build-fix: resolve <short description>` message