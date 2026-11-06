# Planner Agent – NITA Project
**Role**: Decompose feature requests into TDD workflow steps, create the plan artifact, and coordinate between builder and reviewer.

## Initialization (`/ecc:plan "<feature description>"`)
1. **Parse the request** – identify the quant type to add/modify, any CMake flags needed.
2. **Check existing state**:
   - `quant_table.md` – is the quant type already listed?
   - `CMakeLists.txt` – is the `-DQUANT_TYPE` option already defined?
   - `src/q4_matmul.c` – is there already a `QUANT_TYPE == <value>` branch?
3. **Generate the Plan** (editable artifact):
   - List files to modify
   - List new tests to add
   - List implementation steps (RED → GREEN → REFACTOR)
   - Estimated time per phase
4. **Output format** (markdown, saved to `ecc/plan_<timestamp>.md`):
```
# Plan: <feature>

## Quant Type
- `-DQUANT_TYPE=<value>` (e.g., TERNARY, GPTQ)

## Files to Modify
- `quant_table.md` – add entry if new
- `component_spec.md` – update C1‑C8 if needed
- `validation_spec.md` – add runtime check if needed
- `CMakeLists.txt` – ensure `QUANT_TYPE` derivation
- `src/q4_matmul.c` – add `#if QUANT_TYPE == <value>` branch
- `tests/kernel_test.c` – add test function

## TDD Phases
- **RED**: Write test → expect fail
- **GREEN**: Minimal implementation → expect pass
- **REFACTOR**: Optimise / add tests → maintain coverage ≥80%
- **VERIFY**: Coverage report, update docs

## Estimated Effort
- 15 min plan & setup
- 10 min RED (test write)
- 20 min GREEN (kernel body)
- 10 min REFACTOR (extra tests/optimisations)
- Total ~55 min for a new quant type

## Next Step
Hand off to **Builder Agent** to begin RED phase.