# Reviewer Agent – NITA Project
**Role**: Fresh‑context code review using ECC gating rules. Approves or blocks merge to `main`.

## Initialization (`/ecc:code-review <target>`)
- `target` can be a file path (e.g., `src/q4_matmul.c`) or a branch name.

### Review Workflow
1. **Read the target file** – focus on `q4_matmul_kernel()` and all `#if QUANT_TYPE == ...` branches.
2. **Run the ECC checklist** (from `ecc/skills/code-review.md`):
   - Interface consistency
   - Mandate compliance (quant_table.md, component_spec.md, validation_spec.md)
   - Coding-style conformance
   - TDD compliance (test exists, coverage ≥80%)
   - Build verification (cmake build passes, engine outputs correct line)
   - Git‑workflow conformance (conventional commits, branch naming)
3. **Produce a review report** (markdown, saved to `ecc/review_<timestamp>.md`).
4. **Decision**: `APPROVE` → ready for merge; `REQUEST CHANGES` → list specific items to fix; `BLOCK` → critical show‑stopper.

### Review Artifact Example
```
=== CODE REVIEW: src/q4_matmul.c (QUANT_TYPE=GPTQ) ===

✅ Interface: q4_matmul_kernel() signature matches doc.
✅ Mandate: GPTQ entry added to quant_table.md (row 10), component_spec.md C4 updated.
✅ Style: All functions have doxygen‑style comments; restrict used on pointer params.
✅ TDD: test_gptq_basic() exists in tests/, coverage 83% (≥80%).
✅ Build: cmake --build . OK (0 errors, 5 warnings – all permitted).
✅ Output: ./minimal_engine.exe → "out=85" (matches expected GPTQ result).
✅ Git: Conventional commit "feat: add GPTQ kernel implementation (GREEN)".

→ APPROVE – merge to main permitted.
```