#!/usr/bin/env python3
"""Interleaved tile-width sampling for the bimodality question.

Runs the same prefill workload at several tile widths in alternating rounds.
If the slow runs line up across widths, the disturbance is external to the
kernel; if only one width goes slow, it is specific to that configuration.
"""

import os
import re
import subprocess
import sys

MODEL = "H:/OLLAMA-Models/GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf"
EXE = "./build/nita.exe"
ROUNDS = 3
REPS = 4
WIDTHS = [8, 16, 32, 48]

rows = []
for rnd in range(ROUNDS):
    for t in WIDTHS:
        env = dict(os.environ, NITA_TILE=str(t))
        p = subprocess.run([EXE, "--model", MODEL, "--prefill", "512", "--reps", str(REPS)],
                           capture_output=True, text=True, env=env)
        vals = [float(m) for m in re.findall(r"run \d+: [\d.]+ s  ([\d.]+) ms/token", p.stdout)]
        if len(vals) != REPS:
            sys.stderr.write("unexpected output for tile %d\n%s\n" % (t, p.stdout[-400:]))
            continue
        for v in vals:
            rows.append((rnd, t, v))
        print("round %d tile %-3d %s" % (rnd, t, " ".join("%.2f" % v for v in vals)), flush=True)

print()
print("%-6s %-8s %-8s %-8s %-8s" % ("tile", "median", "min", "max", "spread%"))
for t in WIDTHS:
    v = sorted(x[2] for x in rows if x[1] == t)
    if not v:
        continue
    med = v[len(v) // 2]
    print("%-6d %-8.2f %-8.2f %-8.2f %-8.1f" % (t, med, v[0], v[-1], 100.0 * (v[-1] - v[0]) / med))

print()
print("slow-run alignment: for each round, the max across widths and which tile hit it")
for rnd in range(ROUNDS):
    per = {t: max(x[2] for x in rows if x[0] == rnd and x[1] == t) for t in WIDTHS}
    worst = max(per, key=per.get)
    print("  round %d: %s  worst=tile %d" % (rnd, " ".join("%d:%.2f" % (t, per[t]) for t in WIDTHS), worst))
