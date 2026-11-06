# Git Workflow – NITA / LLVM‑IR Project
**Always apply** (from `ecc/rules/common/git-workflow.md`)

## 1. Branching Model
- `main` branch: **always** reflects a **working, built** binary
- Feature branches: `feature/<quant-type>` (e.g., `feature/gptq`, `feature/ternary`)
- Hotfix branches: `hotfix/<issue>` for compile or runtime fixes
- Release branches: `release/<version>` when tagging a release

## 2. Commit Messages
Format (conventional commits):
```
<type>(<scope>): <subject>

<body>

<footer>
```
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(q4_matmul): add GPTQ kernel implementation

- Add q4_matmul_impl_gptq() function
- Wire QUANT_TYPE=GPTQ → dispatch in q4_matmul_kernel()
- Update CMakeLists QUANT_TYPE derivation

Signed-off-by: Your Name <you@example.com>
```

## 3. Pull Request Process
1. **Branch** from `main`, work in topic branch
2. **Commit** with conventional messages
3. **Push** branch; open PR against `main`
4. **Require** at least one review from `ecc/agents/reviewer` (or equivalent)
5. **CI must pass**: `cmake --build . && ./minimal_engine.exe` outputs expected result
6. **Squash and merge** into `main` on approval

## 4. Tagging Releases
- `git tag -a v1.0.0 -m "NITA v1.0.0 – Q4_K_M baseline"`
- `git push origin v1.0.0`
- Update `README.md` and `quant_table.md` per release notes

## 4. Git‑Ignored Files
- `build/` directory (CMake cache, ninja files, binaries)
- `.env`, any secret files
- `*.o`, `*.obj` intermediate objects (unless debugging)