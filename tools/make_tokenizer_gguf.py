#!/usr/bin/env python3
"""Create tokenizer-only GGUF files for testing NITA's encoders.

Writes two files plus a JSON sidecar of the same vocabulary/merges so an
independent reference implementation can compute expected token ids:

    tok_spm.gguf   tokenizer.ggml.model = "llama"  (sentencepiece, scores)
    tok_bpe.gguf   tokenizer.ggml.model = "gpt2"   (byte-level BPE, merges)

Usage:
    python3 tools/make_tokenizer_gguf.py build/
"""

import json
import os
import struct
import sys

GGUF_MAGIC = b"GGUF"
VERSION = 3
ALIGNMENT = 32

UINT32, INT32, FLOAT32, STRING, ARRAY = 4, 5, 6, 8, 9

SP = "\u2581"


def gstr(t: str) -> bytes:
    b = t.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_str(k, v):
    return gstr(k) + struct.pack("<I", STRING) + gstr(v)


def kv_i32(k, v):
    return gstr(k) + struct.pack("<I", INT32) + struct.pack("<i", v)


def kv_u32(k, v):
    return gstr(k) + struct.pack("<I", UINT32) + struct.pack("<I", v)


def kv_arr_str(k, values):
    out = gstr(k) + struct.pack("<I", ARRAY)
    out += struct.pack("<I", STRING) + struct.pack("<Q", len(values))
    for v in values:
        out += gstr(v)
    return out


def kv_arr_f32(k, values):
    out = gstr(k) + struct.pack("<I", ARRAY)
    out += struct.pack("<I", FLOAT32) + struct.pack("<Q", len(values))
    for v in values:
        out += struct.pack("<f", float(v))
    return out


def write_gguf(path, kv_entries):
    header = GGUF_MAGIC
    header += struct.pack("<I", VERSION)
    header += struct.pack("<Q", 0)                 # tensor count
    header += struct.pack("<Q", len(kv_entries))   # metadata count

    body = header + b"".join(kv_entries)
    pad = (-len(body)) % ALIGNMENT
    body += b"\x00" * pad

    with open(path, "wb") as f:
        f.write(body)


# ----------------------------------------------------------------------
# sentencepiece vocabulary
# ----------------------------------------------------------------------
SPM_WORDS = [
    "<unk>", "<s>", "</s>",
    SP + "hello", SP + "world", SP,
    "h", "e", "l", "o", "w", "r", "d",
    "he", "ll", "lo", "hel", "hell", "llo", "hello",
    "wo", "or", "rl", "ld", "wor", "orl", "rld", "worl", "orld", "world",
]

SPM_BYTES = [f"<0x{b:02X}>" for b in range(256)]


def build_spm(outdir):
    tokens = SPM_WORDS + SPM_BYTES
    scores = [-float(i) for i in range(len(tokens))]
    bob = tokens.index("<s>")
    eos = tokens.index("</s>")
    unk = tokens.index("<unk>")

    kv = [
        kv_str("general.architecture", "llama"),
        kv_str("general.name", "nita-tok-spm"),
        kv_u32("general.alignment", ALIGNMENT),
        kv_arr_str("tokenizer.ggml.tokens", tokens),
        kv_arr_f32("tokenizer.ggml.scores", scores),
        kv_i32("tokenizer.ggml.bos_token_id", bob),
        kv_i32("tokenizer.ggml.eos_token_id", eos),
        kv_i32("tokenizer.ggml.unknown_token_id", unk),
        kv_str("tokenizer.ggml.model", "llama"),
    ]

    path = os.path.join(outdir, "tok_spm.gguf")
    write_gguf(path, kv)
    with open(path + ".json", "w") as f:
        json.dump({"model": "llama", "tokens": tokens, "scores": scores,
                   "bos": bob, "eos": eos, "unk": unk, "merges": []}, f, indent=1)
    print(f"wrote {path} ({len(tokens)} tokens)")


# ----------------------------------------------------------------------
# GPT-2 byte-level vocabulary
# ----------------------------------------------------------------------
def gpt2_byte_to_unicode():
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
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = gpt2_byte_to_unicode()
U2B = {v: k for k, v in B2U.items()}


def esc(text: str) -> str:
    """Encode a python string into the byte-level escaped domain."""
    return "".join(B2U[b] for b in text.encode("utf-8"))


def build_bpe(outdir):
    # All 256 single-byte tokens first, as real GPT-2 vocabularies do.
    tokens = [B2U[b] for b in range(256)]

    # Rank order matters: earlier merges win. Real merge tables build
    # "hello" through the whole chain before ever merging "l l", so keep
    # that ordering here.
    merges = [
        ("\u0120", "t"),
        ("\u0120t", "h"),
        ("\u0120th", "e"),
        ("h", "e"),
        ("he", "l"),
        ("hel", "l"),
        ("hell", "o"),
        ("l", "l"),
    ]

    # Every merge must produce a vocabulary entry.
    for a, b in merges:
        piece = a + b
        if piece not in tokens:
            tokens.append(piece)

    merge_strings = [f"{a} {b}" for a, b in merges]

    kv = [
        kv_str("general.architecture", "llama"),
        kv_str("general.name", "nita-tok-bpe"),
        kv_u32("general.alignment", ALIGNMENT),
        kv_arr_str("tokenizer.ggml.tokens", tokens),
        kv_arr_str("tokenizer.ggml.merges", merge_strings),
        kv_i32("tokenizer.ggml.bos_token_id", -1),
        kv_i32("tokenizer.ggml.eos_token_id", -1),
        kv_i32("tokenizer.ggml.unknown_token_id", -1),
        kv_str("tokenizer.ggml.model", "gpt2"),
    ]

    path = os.path.join(outdir, "tok_bpe.gguf")
    write_gguf(path, kv)
    with open(path + ".json", "w") as f:
        json.dump({"model": "gpt2", "tokens": tokens, "scores": [],
                   "bos": -1, "eos": -1, "unk": -1, "merges": merge_strings}, f, indent=1)
    print(f"wrote {path} ({len(tokens)} tokens, {len(merges)} merges)")


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)
    build_spm(outdir)
    build_bpe(outdir)


if __name__ == "__main__":
    main()
