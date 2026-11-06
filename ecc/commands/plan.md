# /ecc:plan Command – NITA Project
**Use**: Before starting any implementation, to create a structured plan artifact.

### Syntax
```
/ecc:plan "<feature description>"
```

### What It Does
1. Parses the feature description (e.g., "Add GPTQ support", "Add TERNARY kernel").
2. Checks existing state (quant_table.md, CMakeLists.txt, q4_matmul.c).
3. Generates a plan markdown file saved to `ecc/plan_<timestamp>.md`.
4. Outputs a summary to the chat.

### Example
```
User: /ecc:plan "Add GPTQ kernel support"
ECC: 
- Checking quant_table.md... GPTQ already row 10 ✅
- Checking CMakeLists.txt... QUANT_TYPE derivation OK ✅
- Checking q4_matmul.c... GPTQ branch exists ✅
- Generating plan: ecc/plan_20261106_1430.md
- Summary: GPTQ support present; ready for TDD RED/GREEN/REFACTOR cycle.
```