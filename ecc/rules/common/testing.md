# Testing – NITA / LLVM‑IR Project (TDD Workflow)
**Always apply** (from `ecc/rules/common/testing.md`)

## TDD Workflow (RED → GREEN → REFACTOR)

### Step 1 – RED: Define Interfaces & Write Failing Tests
1. **Define the kernel interface** first (function signature, expected inputs/outputs).
2. **Write a minimal test harness** that calls `q4_matmul_kernel()` with known data.
3. **Run the test** – it must **fail** (RED) because the implementation is not yet compiled/linked.
4. **Commit** the failing test to `tests/` with message `test: add minimal kernel harness`.

### Step 2 – GREEN: Implement Minimal Code to Pass
1. **Implement the simplest possible** kernel body that passes the test.
2. **Do not** add optimisations, AVX2 intrinsics, or edge‑case handling yet.
3. **Run the test** – it must **pass** (GREEN).
4. **Commit** the implementation with `feat: add minimal kernel body`.

### Step 3 – REFACTOR: Improve Code Quality
1. **Add AVX2 intrinsics** or optimise the dot‑product while keeping the test passing.
2. **Add additional test cases** (different M,N,K sizes, edge cases).
3. **Run the full test suite** – all must pass.
4. **Commit** with `refactor: improve kernel performance / add test cases`.

### Test Coverage Requirement
- **Minimum 80% line coverage** across all `src/*.c` files.
- Use **GCC `--coverage`** or **Clang coverage** to generate reports.
- Command: `cmake --build . -DCMAKE_BUILD_TYPE=Debug && gcov src/q4_matmul.c src/main.c`
- Commit coverage report or note coverage percentage in PR description.

### Test Harness Skeleton (example)
```c
/* tests/kernel_test.c */
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

/* Include the kernel via the build system */
extern void q4_matmul_kernel(const float *restrict scales,
                              const uint8_t *restrict A,
                              const float *restrict B_scales,
                              const int32_t *restrict B,
                              int32_t *restrict out,
                              int M, int N, int K);

void test_q4_k_m_basic(void)
{
    float scales[1] = {1.0f};
    uint8_t A[2] = {0x55, 0xAA};   /* nibbles 5 and 10 */
    float B_scales[1] = {1.0f};
    int32_t B[1] = {1};
    int32_t out;

    q4_matmul_kernel(scales, A, B_scales, B, &out, 1, 1, 1);
    /* Expected: 5 (nibble 5 * scale 1) */
    assert(out == 5);
    printf("[test] Q4_K_M basic: out=%d (expected 5) – PASS\\n", out);
}

void test_ternary_basic(void)
{
    float scales[1] = {1.0f};
    int8_t A[3] = {1, -1, 0};   /* ternary weights */
    float B_scales[1] = {1.0f};
    int32_t B[1] = {1};
    int32_t out;

    q4_matmul_kernel(scales, (uint8_t *)A, B_scales, B, &out, 1, 1, 3);
    /* Expected: 1 (only the 1 contributes; -1 and 0 are skipped) */
    assert(out == 1);
    printf("[test] Ternary basic: out=%d (expected 1) – PASS\\n", out);
}
```
Run with: `cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON .. && cmake --build . && ./tests/kernel_test`

## Rules Summary
- **Never commit** code that does not have an associated test (or a documented `FIXME` reason).
- **All new quant‑type branches** must have at least one test case in `tests/`.
- **Refactor** is only allowed after the GREEN step and passing coverage threshold.