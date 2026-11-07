#!/usr/bin/env python3
"""Alternate tile 8 and tile 32 back to back.

If the wide tile is slow while the narrow one is fast in the same time window,
the effect is tile-specific. If both move together, it is machine load.
"""

import os
import re
import subprocess

MODEL = "H:/OLLAMA-Models/GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf"
EXE = "./build/nita.exe"
ROUNDS = 8
REPS = 2


def run(tile):
    env = dict(os.environ, NITA_TILE=str(tile))
    p = subprocess.run([EXE, "--model", MODEL, "--prefill", "384", "--reps", str(REPS)],
                       capture_output=True, text=True, env=env)
    vals = [float(m) for m in re.findall(r"run \d+: [\d.]+ s  ([\d.]+) ms/token", p.stdout)]
    tot = [float(m) for m in re.findall(r"\(([\d.]+) ms/token\)", p.stdout)]
    return vals, (tot[0] if tot else float("nan"))


print("%-6s %-22s %-22s" % ("round", "tile 8 runs / median", "tile 32 runs / median"))
narrow, wide = [], []
for r in range(ROUNDS):
    a, am = run(8)
    b, bm = run(32)
    narrow.append(am)
    wide.append(bm)
    print("%-6d %-22s %-22s" % (r,
          "%s / %.2f" % (" ".join("%.2f" % v for v in a), am),
          "%s / %.2f" % (" ".join("%.2f" % v for v in b), bm)))

print()
print("narrow (tile 8)  median %.2f  min %.2f  max %.2f" % (sorted(narrow)[ROUNDS // 2], min(narrow), max(narrow)))
print("wide   (tile 32) median %.2f  min %.2f  max %.2f" % (sorted(wide)[ROUNDS // 2], min(wide), max(wide)))
ratios = [w / n for w, n in zip(wide, narrow)]
print("per-round ratio wide/narrow: " + " ".join("%.2f" % r for r in ratios))
