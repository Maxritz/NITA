# Builder Agent – NITA Project
**Role**: Execute the TDD workflow (RED → GREEN → REFACTOR) for quant‑type kernel implementation. Writes minimal kernel body, runs build, verifies output.

## Initialization (`/ecc:builder "<phase>"`)
- Phase must be one of: `RED`, `GREEN`, `REFACTOR`, `VERIFY`
- Builder reads the current `QUANT_TYPE` from CMake configuration.

### RED Phase – Write Failing Test
1. **Create/extend** `tests/kernel_test.c` with a test function for the quant type.
   - Example: `test_gptq_basic()` calling `q4_matmul_kernel()` with known data.
2. **Build**: `cmake --build .`
3. **Run**: `./tests/kernel_test` (or `./minimal_engine.exe` if test is embedded).
4. **Verify**: Output shows FAIL (RED) or error list.
5. **Commit**: `test: add <quant> kernel harness (RED)`.

### GREEN Phase – Minimal Implementation
1. **Locate** the `#if QUANT_TYPE == <value>` block in `src/q4_matmul.c`.
2. **Write the simplest possible kernel body** that makes the RED test pass.
   - Use scalar dot‑product pattern (already in default Q4_K_M branch).
   - Do NOT add AVX2 intrinsics yet; keep it portable.
3. **Build**: `cmake --build .`
4. **Run**: `./minimal_engine.exe` – must print the expected line.
4. **Commit**: `feat: add minimal <quant> kernel body (GREEN)`.

### REFACTOR Phase – Optimise & Expand
1. **Add optimisations**: AVX2 intrinsics, loop unrolling, tiling (optional).
2. **Add test cases**: larger M,N,K dimensions, edge cases (K=0, etc.).
3. **Run coverage**: `gcov src/q4_matmul.c` – ensure ≥80%.
4. **Run full test suite** – all pass.
5. **Commit**: `refactor: improve <quant> kernel / add test cases + coverage`.

### VERIFY Phase – Final Checks
1. **Run coverage**: `cmake -DCMAKE_BUILD_TYPE=Debug . && cmake --build . && gcov src/q4_matmul.c src/main.c`
2. **Report percentage** – must be ≥80%.
3. **Update documentation**:
   - `quant_table.md` – confirm entry.
   - `README.md` – reflect new quant type status.
4. **Hand off to Reviewer Agent**.

### Example Session
```
/ecc:builder GREEN
[READ] current QUANT_TYPE=GPTQ from CMake cache
[WRITE] tests/kernel_test.c :: test_gptq_basic() – calls q4_matmul_kernel with 1×1×1 GPTQ weights
[BUILD] cmake --build .
[RUN] ./tests/kernel_test → FAIL (RED) – test not yet passing
WAIT – user said GREEN, not RED. Let’s skip RED.
[WRITE] #if QUANT_TYPE == GPTQ block in q4_matmul.c with scalar loop
[BUILD] cmake --build .
[RUN] ./minimal_engine.exe → "out=85" (expected for GPTQ test matrix)
[COMMIT] feat: add minimal GPTQ kernel body (GREEN)
[VERIFY] coverage 83% → OK
→ Hand off to /ecc:code-review