# Coding Style – NITA / LLVM‑IR Project
**Always apply** (from `ecc/rules/common/coding-style.md`)

## 1. File Organization
- Source files live under `src/`
- Specification files (markdown mandates, component specs, validation) live at project root
- ECC configuration lives under `ecc/`
- Build artifacts go to `build/` (git-ignored)

## 2. Naming Conventions
- CMake options: ALL_UPPER_SNAKE_CASE, prefixed with `-D`
- Source files: lowercase with hyphens or underscores (e.g., `q4_matmul.c`, `main-c.c`)
- Type macros: `QUANT_TYPE=Q4_K_M` etc.
- Function names: `lowercase_with_underscores` (e.g., `q4_matmul_kernel`)

## 3. C / C++ Style
- Use **GCC/MinGW** style (Strawberry Perl GCC 13.2.0 tested)
- Header guards: `#pragma once` or `FILE_H guard`
- Limit line length to 120 characters
- K&R brace style for functions
- `static` for file‑internal functions
- `restrict` keyword for pointer parameters (as in the kernel)

## 4. Preprocessor / CMake
- Exactly **one** `QUANT_TYPE` macro must be defined at configure time
- `QUANTIZED_GEMV` is **required** (base flag)
- Each quant type flag (`-DTERNARY`, `-DGPTQ`, etc.) maps to a `QUANT_TYPE` value
- `MXFP8` forces a compile‑time error: `#error "MXFP8 not supported on AVX2"`
- CMake options are derived from the `QUANT_TYPE` macro (see `CMakeLists.txt`)

## 4. Comments & Documentation
- Every function must have a **doxygen‑style** comment block (or single‑line `//` for inline)
- Markdown specification files (`quant_table.md`, `component_spec.md`, `validation_spec.md`) are the **authoritative** source for validity rules
- In‑code assertions must match the validation predicates in `validation_spec.md`

## 5. Error Handling
- Use `static_assert` for compile‑time checks (see `validation_spec.md`)
- Runtime `validate_engine_components()` must abort on failure (`assert()`)
- All kernel branches must handle the full `M × N × K` dimensionality (even if stubbed)
- No silent failures: every `printf`/`fprintf` error path must flush/stderr

## 6. Build System
- CMake 3.20+ 
- Generator: **Ninja** (or Make)
- Compile flags: `-march=znver3 -mavx2` (AMD Zen3 AVX2 only)
- No AVX-512, no VNNI, no runtime de‑quant of full f32 weight matrix
- Enable/select quant type via `-DQUANT_TYPE=<value>`