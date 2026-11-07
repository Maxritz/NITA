#!/usr/bin/env python3
"""Compare NITA's qwen35 taps against the PrismML reference dumps.

Both sides write one line per tagged tensor: "<name> <count> v0 v1 ...".
Produce them with

    NITA_Q35_DUMP=mine.txt  ./build/nita.exe --model M.gguf --tokens 5
    NITA_FORK_DUMP=fork.txt ./build/llama_logits_fork.exe M.gguf 5 f32kv

then

    python3 tools/compare_taps.py mine.txt fork.txt

Lines tagged "<name> (view) ..." are ggml's auto-named views; their second
field is not a count, so they are skipped. When a name repeats (a graph can be
executed more than once) the last occurrence wins on both sides.
"""

import sys

import numpy as np


def load(path):
    taps = {}
    for line in open(path, errors="replace"):
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            count = int(fields[1])
        except ValueError:
            continue
        values = np.array([float(x) for x in fields[2:]], dtype=np.float64)
        if values.size != count:
            continue
        taps[fields[0]] = values
    return taps


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip())
        return 2

    mine, ref = load(sys.argv[1]), load(sys.argv[2])

    print("%-26s %7s %9s %9s" % ("tap", "n", "cos", "ratio"))
    for name in sorted(ref, key=lambda s: (s.rsplit("-", 1)[-1], s)):
        if name not in mine:
            continue
        a, b = mine[name], ref[name]
        if a.size != b.size:
            print("%-26s shape mismatch: %d vs %d" % (name, a.size, b.size))
            continue
        print("%-26s %7d %9.5f %9.4f" % (name, a.size, cos(a, b),
                                         np.linalg.norm(a) / np.linalg.norm(b)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
