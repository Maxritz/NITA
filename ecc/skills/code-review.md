# Code‑Review Skill – NITA Project
**Purpose**: Fresh‑context review of changes using ECC gating rules.

## Command: `/ecc:code-review <file-or-branch>`
**When to use**: After implementing a new quant‑type, after refactoring, or before merging a PR.

### Review Checklist
1. **Interface Consistency**
   - Does `q4_matmul_kernel()` signature match the documented form?
   - Are all `QUANT_TYPE` branches present and non‑overlapping?
   - Is `QUANTIZED_GEMV` still enabled?

2. **Mandate Compliance**
   - Check `quant_table.md`: does the new quant type belong in the allowed set (Q1‑K → Q8‑0, Binary, Ternary, GPTQ, AWQ, conditional BF16/FP16)?
   - Check `component_spec.md`: are the C1‑C8 predicates still satisfied?
   - Check `validation_spec.md`: does `validate_engine_components()` still pass?

3. **Coding‑Style Conformance**
   - `ecc/rules/common/coding-style.md` rules followed (naming, file org, comments, error handling).
   - `#pragma once` or guard present.
   - `restrict` keyword used on pointer params.
   - `static_assert` matches `validation_spec.md` predicates.

4. **TDD Compliance**
   - Does the change have an associated test in `tests/`?
   - Is line coverage ≥ 80% after the change?
   - Was the RED → GREEN → REFACTOR cycle followed?

5. **Build Verification**
   - `cmake --build .` passes (zero errors).
- `./minimal_engine.exe` outputs the correct line for the selected `QUANT_TYPE`.

6. **Git‑Workflow Conformance**
   - Conventional commit message format.
   - Branch named `feature/<quant-type>` or `hotfix/<issue>`.
   - PR follows the process in `ecc/rules/common/git-workflow.md`.

### Example Review Output
```
=== CODE REVIEW: src/q4_matmul.c ===

✅ Interface: q4_matmul_kernel() signature matches doc.
✅ Mandate: QUANT_TYPE=GPTQ added to quant_table.md, component_spec.md updated.
✅ Style: All functions have doxygen‑style comments; restrict used.
✅ TDD: test_gptq_basic() exists, coverage 83%.
✅ Build: cmake --build . OK; ./minimal_engine.exe outputs "out=85".
✅ Git: Conventional commit "feat: add GPTQ kernel implementation (GREEN)".

→ APPROVE – ready for merge to main.
```