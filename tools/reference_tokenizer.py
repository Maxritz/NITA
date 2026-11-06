#!/usr/bin/env python3
"""Independent reference tokenizers plus a comparison harness for NITA.

Implements the same two encoders as ``src/tokenizer.c`` from their
specifications (they share no code with the C implementation):

  * sentencepiece: spaces become U+2581, then adjacent symbols merge by
    descending token score.
  * GPT-2: bytes map through the byte<->unicode table, then adjacent symbols
    merge in ascending rank order using the merges list.

Usage:
    python3 tools/reference_tokenizer.py --gguf build/tok_spm.gguf \
        --json build/tok_spm.gguf.json --exe build/nita.exe --text "hello world"
"""

import argparse
import json
import subprocess
import sys

SP = "\u2581"


def gpt2_tables():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    b2u = {b: chr(c) for b, c in zip(bs, cs)}
    u2b = {v: k for k, v in b2u.items()}
    return b2u, u2b


B2U, U2B = gpt2_tables()


def build_vocab(tokens):
    vocab = {}
    for i, t in enumerate(tokens):
        vocab.setdefault(t, i)
    return vocab


def encode_spm(text, tokens, scores):
    vocab = build_vocab(tokens)
    escaped = SP + text.replace(" ", SP)

    parts = list(escaped)
    alive = [True] * len(parts)

    while True:
        best = None
        prev = -1
        for i, ok in enumerate(alive):
            if not ok:
                continue
            if prev >= 0:
                joined = parts[prev] + parts[i]
                idx = vocab.get(joined)
                if idx is not None:
                    score = scores[idx]
                    if (best is None or score > best[0]
                            or (score == best[0] and prev < best[1])):
                        best = (score, prev, i)
            prev = i
        if best is None:
            break
        _, a, b = best
        parts[a] = parts[a] + parts[b]
        alive[b] = False

    ids = []
    for i, ok in enumerate(alive):
        if not ok:
            continue
        piece = parts[i]
        idx = vocab.get(piece)
        if idx is not None:
            ids.append(idx)
        else:
            for byte in piece.encode("utf-8"):
                ids.append(vocab.get(f"<0x{byte:02X}>", 0))
    return ids


def encode_bpe(text, tokens, merges):
    vocab = build_vocab(tokens)
    rank = {m: i for i, m in enumerate(merges)}

    escaped = "".join(B2U[b] for b in text.encode("utf-8"))
    parts = list(escaped)
    alive = [True] * len(parts)

    while True:
        best = None
        prev = -1
        for i, ok in enumerate(alive):
            if not ok:
                continue
            if prev >= 0:
                r = rank.get(parts[prev] + " " + parts[i])
                if r is not None:
                    if (best is None or r < best[0]
                            or (r == best[0] and prev < best[1])):
                        best = (r, prev, i)
            prev = i
        if best is None:
            break
        _, a, b = best
        parts[a] = parts[a] + parts[b]
        alive[b] = False

    ids = []
    for i, ok in enumerate(alive):
        if not ok:
            continue
        piece = parts[i]
        idx = vocab.get(piece)
        if idx is not None:
            ids.append(idx)
        else:
            for ch in piece:
                byte = U2B[ch]
                ids.append(vocab.get(B2U[byte], 0))
    return ids


def reference_ids(meta, text):
    if meta["model"] == "gpt2":
        return encode_bpe(text, meta["tokens"], meta["merges"])
    return encode_spm(text, meta["tokens"], meta["scores"])


def c_ids(exe, gguf, text):
    proc = subprocess.run([exe, "--model", gguf, "--tokenize", text],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "engine error")
    for line in proc.stdout.splitlines():
        if line.startswith("ids"):
            body = line.split(":", 1)[1].strip()
            return [int(x) for x in body.split()] if body else []
    raise RuntimeError("no ids line in engine output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="./nita.exe")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--text", action="append", default=[])
    args = ap.parse_args()

    with open(args.json) as f:
        meta = json.load(f)

    texts = args.text or ["hello", "hello world", "world", "A", "hello!"]

    failures = 0
    for text in texts:
        want = reference_ids(meta, text)
        got = c_ids(args.exe, args.gguf, text)
        ok = want == got
        if not ok:
            failures += 1
        print(f"{'PASS' if ok else 'FAIL'} {text!r:24} c={got} ref={want}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
