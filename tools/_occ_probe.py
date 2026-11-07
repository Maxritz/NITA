#!/usr/bin/env python3
"""Is the idle thread-time recoverable?

Two measurements, interleaved so ambient load hits every arm equally:
  1. spin sweep - A/B the block/wake policy and see whether any setting wins
  2. scaling    - 1/2/4/8/16 threads, to see how much of the run is parallel
                  at all (parallel efficiency = t1 / (n * tn))
"""

import os
import re
import statistics
import subprocess

MODEL = "H:/OLLAMA-Models/GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf"
EXE = "./build/nita.exe"
N = 128
REPS = 2


def run(env_extra):
    env = dict(os.environ, **env_extra)
    p = subprocess.run([EXE, "--model", MODEL, "--prefill", str(N),
                        "--reps", str(REPS)],
                       capture_output=True, text=True, env=env)
    m = re.search(r"\(([\d.]+) ms/token\).*\(median\)", p.stdout)
    cpu = re.findall(r"cpu ([\d.]+)", p.stdout)
    if not m:
        raise SystemExit("bad output:\n" + p.stdout[-400:])
    return float(m.group(1)), (statistics.mean(float(c) for c in cpu) if cpu else 0.0)


print("=== spin sweep (interleaved) ===")
arms = {"1000": {"NITA_SPIN": "1000"},
        "10000": {"NITA_SPIN": "10000"},
        "100000": {"NITA_SPIN": "100000"}}
acc = {k: [] for k in arms}
util = {k: [] for k in arms}
for r in range(5):
    for k, e in arms.items():
        ms, cpu = run(e)
        acc[k].append(ms)
        util[k].append(cpu)
    print("  round %d: %s" % (r + 1, "  ".join(
        "%s %.2f" % (k, acc[k][-1]) for k in arms)), flush=True)

print()
print("%-8s %7s %7s %7s %8s" % ("spin", "best", "median", "worst", "cpu avg"))
for k in arms:
    print("%-8s %7.2f %7.2f %7.2f %7.1f/16"
          % (k, min(acc[k]), statistics.median(acc[k]), max(acc[k]),
             statistics.mean(util[k])))

print()
print("=== thread scaling (same interleaving) ===")
th = [1, 2, 4, 8, 16]
res = {t: [] for t in th}
for r in range(3):
    for t in th:
        ms, cpu = run({"NITA_THREADS": str(t)})
        res[t].append(ms)
    print("  round %d: %s" % (r + 1, "  ".join(
        "t%d %.1f" % (t, res[t][-1]) for t in th)), flush=True)

print()
base = statistics.median(res[1])
print("%-6s %8s %10s %10s" % ("threads", "ms/tok", "speedup", "efficiency"))
for t in th:
    ms = statistics.median(res[t])
    print("%-6d %8.2f %10.2f %9.0f%%" % (t, ms, base / ms, 100.0 * base / (ms * t)))
