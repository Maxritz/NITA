# TDD Workflow Skill – NITA Project
**Purpose**: Provide a structured Test‑Driven‑Development loop for the NITA LLVM‑IR inference engine.

## Command: `/ecc:tdd <description>`
**When to use**: Starting a new quant‑type implementation, adding a new test, or fixing a regression.

### Phase 1 – PLAN (1‑2 minutes)
1. `/ecc:plan "Add TERNARY kernel test harness"` – capture the plan as an editable artifact.
2. Identify the **interface**: `void q4_matmul_kernel(..., int M, int N, int K)`.
3. Identify the **test data** (small matrices, known expected outputs).

### Phase 2 – RED (3‑5 minutes)
1. **Write the test** in `tests/` directory.
   - Example: `tests/kernel_test.c` with `test_ternary_basic()`.
   - The test must call `q4_matmul_kernel()` with stub data.
2. **Build** the project: `cmake --build .`
3. **Run the test** – it must **FAIL** (RED phase).
4. **Commit**: `test: add TERNARY kernel harness (RED)`.

### Phase 3 – GREEN (5‑10 minutes)
1. **Implement the minimal kernel body** that makes the test pass.
   - Do not add optimisations, AVX2 intrinsics, or edge cases yet.
   - Use the existing scalar dot‑product pattern from `q4_matmul.c`.
2. **Build** and **run** the test – it must **PASS** (GREEN).
3. **Commit**: `feat: add minimal TERNARY kernel body (GREEN)`.

### Phase 4 – REFACTOR (3‑5 minutes)
1. **Add optimisations** or expand the implementation (AVX2 intrinsics, larger test matrices).
2. **Add more test cases** (different M,N,K sizes, edge cases).
3. **Run the full test suite** – all must pass.
4. **Run coverage**: `cmake -DCMAKE_BUILD_TYPE=Debug . && cmake --build . && gcov src/q4_matmul.c`
5. **Commit**: `refactor: improve TERNARY kernel / add test cases + coverage report`.

### Phase 5 – VERIFY (1 minute)
1. Ensure **80%+ line coverage** across `src/*.c`.
2. Update `README.md` with the new quant‑type status.
3. Update `quant_table.md` if the new type changes the mandate table.
4. **Close** the TDD cycle.

### Example `/ecc:tdd` Session Log
```
[PLAN] /ecc:plan "Add GPTQ kernel test harness"
[RED]   Write tests/kernel_test.c :: test_gptq_basic() – builds & runs → FAIL
[GREEN] Implement q4_matmul_impl_gptq() scalar loop – builds & runs → PASS
[REF]   Add test_gptq_edge_case() 512‑element matrix – builds & runs → PASS
[VER]   Coverage 82% – update README.md, quant_table.md if needed
```