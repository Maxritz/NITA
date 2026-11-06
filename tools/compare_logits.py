#!/usr/bin/env python3
"""Compare NITA's C forward pass against the numpy reference.

Runs ``nita --model <gguf> --tokens a,b,c`` and the reference implementation
on the same ids, then prints the maximum absolute difference.

Exit code is 0 when every logit agrees within the tolerance (default 1e-4).
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_forward  # noqa: E402


def c_logits(exe, gguf, ids):
    proc = subprocess.run(
        [exe, "--model", gguf, "--tokens", ",".join(str(i) for i in ids)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "engine exited with an error")

    for line in proc.stdout.splitlines():
        if line.startswith("logits:"):
            return [float(x) for x in line.split()[1:]]
    raise RuntimeError("no logits line in engine output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="./nita.exe")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    ids = [int(t) for t in args.tokens.split(",") if t.strip() != ""]

    got = c_logits(args.exe, args.gguf, ids)

    data = np.load(args.npz)
    w = {k: data[k].astype(np.float32) for k in data.files}
    ref, _ = reference_forward.forward(w, ids)

    if len(got) != len(ref):
        print(f"FAIL: length mismatch (c={len(got)} ref={len(ref)})")
        return 1

    diff = np.abs(np.asarray(got, dtype=np.float64) - np.asarray(ref, dtype=np.float64))
    worst = float(diff.max())

    status = "PASS" if worst <= args.tol else "FAIL"
    print(f"{status} tokens={ids} max|delta|={worst:.3e} (tol {args.tol:g})")
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
