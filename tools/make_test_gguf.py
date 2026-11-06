#!/usr/bin/env python3
"""Generate a small, structurally valid GGUF v3 file for NITA testing.

The file is not a usable language model; it exists so the loader, tokenizer
and HTTP layer can be exercised without a multi-gigabyte download.
"""

import struct
import sys

GGUF_MAGIC = b"GGUF"
VERSION = 3

# metadata value types
UINT32 = 4
INT32 = 5
FLOAT32 = 6
STRING = 8
ARRAY = 9

ALIGNMENT = 32


def s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key: str, value: str) -> bytes:
    return s(key) + struct.pack("<I", STRING) + s(value)


def kv_int32(key: str, value: int) -> bytes:
    return s(key) + struct.pack("<I", INT32) + struct.pack("<i", value)


def kv_uint32(key: str, value: int) -> bytes:
    return s(key) + struct.pack("<I", UINT32) + struct.pack("<I", value)


def kv_array_string(key: str, values) -> bytes:
    out = s(key) + struct.pack("<I", ARRAY)
    out += struct.pack("<I", STRING) + struct.pack("<Q", len(values))
    for v in values:
        out += s(v)
    return out


def kv_array_f32(key: str, values) -> bytes:
    out = s(key) + struct.pack("<I", ARRAY)
    out += struct.pack("<I", FLOAT32) + struct.pack("<Q", len(values))
    for v in values:
        out += struct.pack("<f", v)
    return out


def main(path: str) -> None:
    vocab = [
        "<unk>", "<s>", "</s>",
        "\u2581hello", "\u2581world", "\u2581there",
        "\u2581the", "\u2581a", "\u2581is",
        ".", ",", "!",
    ]
    scores = [0.0, 0.0, 0.0] + [-float(i) for i in range(1, len(vocab) - 2)]

    kv_entries = [
        kv_string("general.architecture", "llama"),
        kv_string("general.name", "nita-test-model"),
        kv_uint32("general.alignment", ALIGNMENT),
        kv_int32("llama.context_length", 512),
        kv_int32("llama.embedding_length", 8),
        kv_int32("llama.block_count", 1),
        kv_array_string("tokenizer.ggml.tokens", vocab),
        kv_array_f32("tokenizer.ggml.scores", scores),
        kv_int32("tokenizer.ggml.bos_token_id", 1),
        kv_int32("tokenizer.ggml.eos_token_id", 2),
        kv_int32("tokenizer.ggml.unknown_token_id", 0),
        kv_string("tokenizer.ggml.model", "llama"),
    ]
    metadata = b"".join(kv_entries)

    # One F32 tensor: token_embd.weight with dims [8, 12] (8 floats per row)
    n_dims = 2
    dims = [8, len(vocab)]
    tensor_type = 0  # F32
    n_elements = dims[0] * dims[1]
    payload = struct.pack("<" + "f" * n_elements, *[0.01 * i for i in range(n_elements)])

    tensors = b""
    tensors += s("token_embd.weight")
    tensors += struct.pack("<I", n_dims)
    for d in dims:
        tensors += struct.pack("<Q", d)
    tensors += struct.pack("<I", tensor_type)
    tensors += struct.pack("<Q", 0)  # offset into data section

    header = GGUF_MAGIC
    header += struct.pack("<I", VERSION)
    header += struct.pack("<Q", 1)          # tensor count
    header += struct.pack("<Q", len(kv_entries))  # metadata count

    body = header + metadata + tensors
    pad = (-len(body)) % ALIGNMENT
    body += b"\x00" * pad
    body += payload

    with open(path, "wb") as f:
        f.write(body)

    print(f"wrote {path} ({len(body)} bytes, {len(vocab)} vocab entries)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "test_model.gguf"
    main(out)
