#!/usr/bin/env python3
"""Check NITA's Bonsai logits against the PrismML fork, in one command.

This is the machine-checkable form of the tap comparison: it runs both
runtimes on the same tokens, dumps every tagged tensor, and fails if the
logits or any tap drifts. Wire it as a CTest entry with

    cmake -S . -B build -DNITA_TEST_MODEL=M.gguf \
          -DNITA_FORK_LOGITS=build/llama_logits_fork.exe

or run it by hand:

    tools/check_fork_match.py --engine build/nita --fork build/llama_logits_fork.exe \\
                              --model M.gguf --tokens 5,6,7

Two things about the dumps it has to reconcile. The engine writes one line per
tag per token, so a tag appears `n_tokens` times; ggml keeps the token axis in
the tensor, so the fork writes one line holding every token back to back, and
for a few taps (the last block, the final norm) it only keeps the last token.
Tokens are therefore compared from the end, as many as both sides have.

Standard library only: this runs under whichever interpreter CMake found, and
that one is not guaranteed to have numpy.

Exit status is 0 for a match, 1 for a mismatch, and 77 when a prerequisite is
missing (CTest reads 77 as "skipped").
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile

SKIP = 77


def load_dumps(path):
    """name -> list of vectors, one per written line, in file order.

    ggml renames views to "<name> (view)"; their second field is not a count,
    so they are dropped here rather than treated as a tap.
    """
    taps = {}
    for line in open(path, errors="replace"):
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            count = int(fields[1])
        except ValueError:
            continue
        if len(fields) - 2 != count:
            continue
        try:
            values = [float(x) for x in fields[2:]]
        except ValueError:
            continue
        taps.setdefault(fields[0], []).append(values)
    return taps


def logits_of(path):
    for line in open(path, errors="replace"):
        if not line.startswith("logits:"):
            continue
        try:
            return [float(x) for x in line.split()[1:]]
        except ValueError:
            return None
    return None


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def cos(a, b):
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def top_k(values, k=5):
    order = sorted(range(len(values)), key=lambda i: -values[i])
    return order[:k]


def rows_of(values, width):
    return [values[i:i + width] for i in range(0, len(values), width)]


def run(cmd, env, log_path):
    """Run with stderr kept out of stdout: llama.cpp logs while printing the
    logits line, and an interleaved line cannot be parsed."""
    with open(log_path, "w") as out, open(log_path + ".err", "w") as err:
        try:
            return subprocess.call(cmd, env=env, stdout=out, stderr=err)
        except OSError as exc:
            out.write("cannot run %s: %s\n" % (cmd[0], exc))
            return 127


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", required=True, help="engine binary")
    ap.add_argument("--fork", required=True, help="fork logits driver")
    ap.add_argument("--model", required=True, help="Bonsai checkpoint")
    ap.add_argument("--tokens", default="5,6,7", help="token ids, comma separated")
    ap.add_argument("--min-logit-cos", type=float, default=0.9999)
    ap.add_argument("--min-tap-cos", type=float, default=0.995)
    ap.add_argument("--keep-dumps", metavar="DIR",
                    help="write the dumps here instead of a temporary directory")
    args = ap.parse_args()

    # Windows CreateProcess does not resolve a relative "build/nita.exe" in
    # some Python builds, so settle every path once, up front.
    args.engine = os.path.abspath(args.engine)
    args.fork = os.path.abspath(args.fork)
    args.model = os.path.abspath(args.model)

    for path in (args.engine, args.fork, args.model):
        if not os.path.exists(path):
            print("skipped: %s does not exist" % path)
            return SKIP

    work = args.keep_dumps or tempfile.mkdtemp(prefix="nita-fork-")
    os.makedirs(work, exist_ok=True)
    mine_dump = os.path.join(work, "mine_taps.txt")
    fork_dump = os.path.join(work, "fork_taps.txt")
    mine_out = os.path.join(work, "mine.txt")
    fork_out = os.path.join(work, "fork.txt")

    env = dict(os.environ)
    env["NITA_Q35_DUMP"] = mine_dump
    rc = run([args.engine, "--model", args.model, "--tokens", args.tokens],
             env, mine_out)
    if rc != 0:
        print("engine run failed (%d), log: %s" % (rc, mine_out))
        return 1

    env.pop("NITA_Q35_DUMP")
    env["NITA_FORK_DUMP"] = fork_dump
    rc = run([args.fork, args.model, args.tokens, "f32kv"], env, fork_out)
    if rc != 0:
        print("fork run failed (%d), log: %s" % (rc, fork_out))
        return 1

    mine_logits = logits_of(mine_out)
    fork_logits = logits_of(fork_out)
    if mine_logits is None:
        print("engine printed no parseable logits line, log: %s" % mine_out)
        return 1
    if fork_logits is None:
        print("fork printed no parseable logits line, log: %s" % fork_out)
        return 1
    if len(mine_logits) != len(fork_logits):
        print("logit count differs: %d vs %d"
              % (len(mine_logits), len(fork_logits)))
        return 1

    failures = []
    logit_cos = cos(mine_logits, fork_logits)
    mine_top = top_k(mine_logits)
    fork_top = top_k(fork_logits)
    mean_abs = sum(abs(a - b) for a, b in
                   zip(mine_logits, fork_logits)) / len(mine_logits)
    print("logits: %d values, cos %.6f, argmax %d/%d"
          % (len(mine_logits), logit_cos, mine_top[0], fork_top[0]))
    print("        top-5 %s vs %s, mean |diff| %.4f"
          % (mine_top, fork_top, mean_abs))
    if logit_cos < args.min_logit_cos:
        failures.append("logits cos %.6f < %.6f" % (logit_cos, args.min_logit_cos))
    if mine_top[0] != fork_top[0]:
        failures.append("argmax %d != %d" % (mine_top[0], fork_top[0]))
    if set(mine_top) != set(fork_top):
        failures.append("top-5 sets differ: %s vs %s" % (mine_top, fork_top))

    mine, fork = load_dumps(mine_dump), load_dumps(fork_dump)
    compared, worst = 0, []
    for name in sorted(mine):
        if name not in fork:
            continue
        lines = mine[name]
        width = len(lines[-1])
        ref = rows_of(fork[name][-1], width)
        # A tap may only be captured for the final token on the fork side, so
        # compare as many trailing tokens as both sides have.
        shared = min(len(lines), len(ref))
        if shared == 0:
            failures.append("%s: nothing to compare" % name)
            continue
        tap_cos = min(cos(a, b) for a, b in
                      zip(lines[len(lines) - shared:], ref[len(ref) - shared:]))
        compared += 1
        worst.append((tap_cos, name))
        if tap_cos < args.min_tap_cos:
            failures.append("%s: cos %.6f < %.6f" % (name, tap_cos, args.min_tap_cos))

    worst.sort()
    print("taps:   %d compared, worst %s" % (compared, ", ".join(
        "%s %.6f" % (name, c) for c, name in worst[:5]) or "none"))
    if compared == 0:
        failures.append("no tap compared - did the dumps get written?")

    if failures:
        print("\nFAIL")
        for reason in failures:
            print("  " + reason)
        print("\nlogs and dumps in %s" % work)
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
