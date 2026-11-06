#!/usr/bin/env python3
"""Generate a complete, randomly-initialised tiny llama-style GGUF model.

The model is far too small to be useful for language, but it exercises the
whole forward pass: embeddings, RMSNorm, RoPE, grouped-query attention with a
KV cache, SwiGLU feed-forward and the output projection.

Weights are also written to ``<out>.npz`` in the same orientation the C code
sees them, so ``tools/reference_forward.py`` can compute logits independently.

Usage:
    python3 tools/make_tiny_model.py build/tiny.gguf [--dtype f32|q8_0]
"""

import argparse
import struct

import numpy as np

GGUF_MAGIC = b"GGUF"
VERSION = 3
ALIGNMENT = 32

UINT32, INT32, FLOAT32, STRING, ARRAY = 4, 5, 6, 8, 9

GGML_F32 = 0
GGML_Q8_0 = 8
GGML_Q1_0 = 41
GGML_Q2_0 = 42
GGML_PQ2_0 = 142
GGML_PTQ1_0 = 143

VOCAB = [
    "<unk>", "<s>", "</s>",
    "\u2581hello", "\u2581world", "\u2581the", "\u2581a", "\u2581is",
    "\u2581good", "\u2581bad", "\u2581yes", "\u2581no",
    ".", "!", "\u2581ok", "\u2581fine",
]

# Hyperparameters
N_EMBD = 128
N_LAYER = 2
N_HEAD = 4
N_HEAD_KV = 2
N_FF = 256
N_VOCAB = len(VOCAB)
CTX = 64
HEAD_DIM = N_EMBD // N_HEAD          # 32
KV_DIM = HEAD_DIM * N_HEAD_KV        # 64
N_ROT = HEAD_DIM
RMS_EPS = 1e-5
ROPE_THETA = 10000.0


def gstr(t: str) -> bytes:
    b = t.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_str(k, v):
    return gstr(k) + struct.pack("<I", STRING) + gstr(v)


def kv_i32(k, v):
    return gstr(k) + struct.pack("<I", INT32) + struct.pack("<i", v)


def kv_u32(k, v):
    return gstr(k) + struct.pack("<I", UINT32) + struct.pack("<I", v)


def kv_f32(k, v):
    return gstr(k) + struct.pack("<I", FLOAT32) + struct.pack("<f", v)


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


def quantise_q8_0(mat: np.ndarray) -> bytes:
    """Encode a 2-D array as Q8_0 blocks of 32 elements."""
    flat = mat.reshape(-1)
    assert flat.size % 32 == 0, "rows must be a multiple of 32 elements"
    out = bytearray()
    for start in range(0, flat.size, 32):
        blk = flat[start:start + 32]
        amax = float(np.max(np.abs(blk))) if blk.size else 0.0
        d = amax / 127.0 if amax > 0 else 0.0
        inv = 127.0 / amax if amax > 0 else 0.0
        q = np.clip(np.round(blk.astype(np.float64) * inv), -127, 127).astype(np.int8)
        out += np.float16(d).tobytes()
        out += q.tobytes()
    return bytes(out)


def dequantise_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    """Inverse of quantise_q8_0, used to record the effective weights."""
    out = np.empty(n_elements, dtype=np.float32)
    pos = 0
    for blk in range(n_elements // 32):
        d = float(np.frombuffer(data[pos:pos + 2], dtype=np.float16)[0])
        pos += 2
        q = np.frombuffer(data[pos:pos + 32], dtype=np.int8).astype(np.float32)
        pos += 32
        out[blk * 32:(blk + 1) * 32] = q * d
    return out


def quantise_q1_0(mat: np.ndarray) -> bytes:
    """Encode as Q1_0: 128-weight groups, one sign bit per weight, one fp16
    scale per group. Matches quantize_row_q1_0_ref, which uses mean |w|."""
    flat = mat.reshape(-1)
    assert flat.size % 128 == 0, "rows must be a multiple of 128 elements"
    out = bytearray()
    for start in range(0, flat.size, 128):
        blk = flat[start:start + 128]
        d = float(np.mean(np.abs(blk)))
        qs = bytearray(16)
        for j, w in enumerate(blk):
            if w > 0:
                qs[j // 8] |= 1 << (j % 8)
        out += np.float16(d).tobytes()
        out += bytes(qs)
    return bytes(out)


def dequantise_q1_0(data: bytes, n_elements: int) -> np.ndarray:
    """Inverse of quantise_q1_0."""
    out = np.empty(n_elements, dtype=np.float32)
    for b in range(n_elements // 128):
        base = b * 18
        d = float(np.frombuffer(data[base:base + 2], dtype=np.float16)[0])
        qs = data[base + 2:base + 18]
        for j in range(128):
            bit = (qs[j // 8] >> (j % 8)) & 1
            out[b * 128 + j] = d if bit else -d
    return out


def quantise_pq2_0(mat: np.ndarray, group: int = 128) -> bytes:
    """Encode as PQ2_0 (group 128) or Q2_0 (group 64): one fp16 scale per
    group, each ternary weight in a 2-bit slot."""
    flat = mat.reshape(-1)
    assert flat.size % group == 0, f"rows must be a multiple of {group} elements"
    out = bytearray()
    for start in range(0, flat.size, group):
        blk = flat[start:start + group]
        amax = float(np.max(np.abs(blk)))
        d = amax
        inv = 1.0 / d if d > 0 else 0.0
        qs = bytearray(group // 4)
        for j, w in enumerate(blk):
            q = int(np.rint(float(w) * inv)) + 1
            q = max(0, min(3, q))
            qs[j // 4] |= (q & 3) << ((j % 4) * 2)
        out += np.float16(d).tobytes()
        out += bytes(qs)
    return bytes(out)


def dequantise_pq2_0(data: bytes, n_elements: int, group: int = 128) -> np.ndarray:
    """Inverse of quantise_pq2_0 (also used for group-64 Q2_0)."""
    out = np.empty(n_elements, dtype=np.float32)
    nbytes = 2 + group // 4
    for b in range(n_elements // group):
        base = b * nbytes
        d = float(np.frombuffer(data[base:base + 2], dtype=np.float16)[0])
        qs = data[base + 2:base + nbytes]
        for j in range(group):
            q = (qs[j // 4] >> ((j % 4) * 2)) & 3
            out[b * group + j] = float(q - 1) * d
    return out


PTQ1_STAGES = (32, 16, 8)
POW3 = (1, 3, 9, 27, 81, 243)


def quantise_ptq1_0(mat: np.ndarray) -> bytes:
    """Encode as PTQ1_0: 128-weight groups, dense base-3 trits."""
    flat = mat.reshape(-1)
    assert flat.size % 128 == 0, "rows must be a multiple of 128 elements"
    out = bytearray()
    for start in range(0, flat.size, 128):
        blk = flat[start:start + 128]
        amax = float(np.max(np.abs(blk)))
        d = amax
        inv = 1.0 / d if d > 0 else 0.0
        tr = [max(0, min(2, int(np.rint(float(w) * inv)) + 1)) for w in blk]

        qs = bytearray(24)
        qh = bytearray(2)
        # Invert the published decoder: it emits byte (j+m)'s trit n at output
        # position n*c + m, so pack value (xoff + n*c + m) into that slot.
        xoff = 0
        j = 0
        for c in PTQ1_STAGES:
            while j + c <= 24:
                for m in range(c):
                    byte = 0
                    for n in range(5):
                        byte = byte * 3 + tr[xoff + n * c + m]
                    # ceiling-divide by 3^5 so the top trit lands in bit 7
                    qs[j + m] = ((byte * 256 + 242) // 243) & 0xFF
                j += c
                xoff += 5 * c
        for h in range(2):
            byte = 0
            for n in range(4):
                byte = byte * 3 + tr[xoff + n * 2 + h]
            byte = byte * 3
            qh[h] = ((byte * 256 + 242) // 243) & 0xFF

        out += bytes(qs)
        out += bytes(qh)
        out += np.float16(d).tobytes()
    return bytes(out)


def dequantise_ptq1_0(data: bytes, n_elements: int) -> np.ndarray:
    """Inverse of quantise_ptq1_0."""
    out = np.empty(n_elements, dtype=np.float32)
    for b in range(n_elements // 128):
        base = b * 28
        qs = data[base:base + 24]
        qh = data[base + 24:base + 26]
        d = float(np.frombuffer(data[base + 26:base + 28], dtype=np.float16)[0])

        vals = []
        j = 0
        for c in PTQ1_STAGES:
            while j + c <= 24:
                for n in range(5):
                    for m in range(c):
                        q = (qs[j + m] * POW3[n]) & 0xFF
                        xi = ((q * 3) & 0xFFFF) >> 8
                        vals.append(float(xi - 1) * d)
                j += c
        for n in range(4):
            for h in range(2):
                q = (qh[h] * POW3[n]) & 0xFF
                xi = ((q * 3) & 0xFFFF) >> 8
                vals.append(float(xi - 1) * d)

        out[b * 128:(b + 1) * 128] = vals
    return out


def build(dtype: str):
    rng = np.random.default_rng(20261106)
    scale = 0.08
    ternary = dtype in ("ptq1_0", "pq2_0", "q2_0")
    binary = dtype == "q1_0"

    def W(n_out, n_in):
        # Shape (n_out, n_in); ne0 = n_in, ne1 = n_out.
        if binary:
            # Q1_0 has no zero state: every weight is +/-d.
            return (rng.integers(0, 2, size=(n_out, n_in)) * 2 - 1).astype(np.float32)
        if ternary:
            # Exactly representable ternary weights, so any decode error shows up
            # in the logits rather than being absorbed by quantisation noise.
            return rng.integers(-1, 2, size=(n_out, n_in)).astype(np.float32)
        return (rng.standard_normal((n_out, n_in)) * scale).astype(np.float32)

    weights = {}
    weights["token_embd.weight"] = W(N_VOCAB, N_EMBD)
    weights["output_norm.weight"] = np.ones(N_EMBD, dtype=np.float32)
    weights["output.weight"] = W(N_VOCAB, N_EMBD)

    for l in range(N_LAYER):
        weights[f"blk.{l}.attn_norm.weight"] = np.ones(N_EMBD, dtype=np.float32)
        weights[f"blk.{l}.attn_q.weight"] = W(N_HEAD * HEAD_DIM, N_EMBD)
        weights[f"blk.{l}.attn_k.weight"] = W(KV_DIM, N_EMBD)
        weights[f"blk.{l}.attn_v.weight"] = W(KV_DIM, N_EMBD)
        weights[f"blk.{l}.attn_output.weight"] = W(N_EMBD, N_HEAD * HEAD_DIM)
        weights[f"blk.{l}.ffn_norm.weight"] = np.ones(N_EMBD, dtype=np.float32)
        weights[f"blk.{l}.ffn_gate.weight"] = W(N_FF, N_EMBD)
        weights[f"blk.{l}.ffn_up.weight"] = W(N_FF, N_EMBD)
        weights[f"blk.{l}.ffn_down.weight"] = W(N_EMBD, N_FF)
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="tiny.gguf")
    ap.add_argument("--dtype", choices=["f32", "q8_0", "q1_0", "q2_0", "pq2_0", "ptq1_0"],
                    default="f32")
    args = ap.parse_args()

    weights = build(args.dtype)

    # Serialise tensors with 32-byte alignment.
    blobs = []
    infos = []
    offset = 0
    effective = {}

    for name, mat in weights.items():
        ne0 = mat.shape[1] if mat.ndim == 2 else mat.shape[0]
        ne1 = mat.shape[0] if mat.ndim == 2 else 1
        dims = [ne0, ne1] if mat.ndim == 2 else [ne0]
        ne0_pad = ne0

        if args.dtype == "q8_0" and ne0_pad % 32 == 0 and ne1 > 1:
            data = quantise_q8_0(mat)
            ttype = GGML_Q8_0
            effective[name] = dequantise_q8_0(data, mat.size).reshape(mat.shape)
        elif args.dtype == "q1_0" and ne0_pad % 128 == 0:
            data = quantise_q1_0(mat)
            ttype = GGML_Q1_0
            effective[name] = dequantise_q1_0(data, mat.size).reshape(mat.shape)
        elif args.dtype == "q2_0" and ne0_pad % 64 == 0:
            data = quantise_pq2_0(mat, 64)
            ttype = GGML_Q2_0
            effective[name] = dequantise_pq2_0(data, mat.size, 64).reshape(mat.shape)
        elif args.dtype == "pq2_0" and ne0_pad % 128 == 0:
            data = quantise_pq2_0(mat)
            ttype = GGML_PQ2_0
            effective[name] = dequantise_pq2_0(data, mat.size).reshape(mat.shape)
        elif args.dtype == "ptq1_0" and ne0_pad % 128 == 0:
            data = quantise_ptq1_0(mat)
            ttype = GGML_PTQ1_0
            effective[name] = dequantise_ptq1_0(data, mat.size).reshape(mat.shape)
        else:
            data = np.ascontiguousarray(mat, dtype=np.float32).tobytes()
            ttype = GGML_F32
            effective[name] = mat.astype(np.float32)

        pad = (-offset) % ALIGNMENT
        if pad:
            blobs.append(b"\x00" * pad)
            offset += pad

        infos.append((name, dims, ttype, offset))
        blobs.append(data)
        offset += len(data)

        pad = (-offset) % ALIGNMENT
        if pad:
            blobs.append(b"\x00" * pad)
            offset += pad

    kv = [
        kv_str("general.architecture", "llama"),
        kv_str("general.name", "nita-tiny"),
        kv_u32("general.alignment", ALIGNMENT),
        kv_i32("llama.context_length", CTX),
        kv_i32("llama.embedding_length", N_EMBD),
        kv_i32("llama.block_count", N_LAYER),
        kv_i32("llama.attention.head_count", N_HEAD),
        kv_i32("llama.attention.head_count_kv", N_HEAD_KV),
        kv_i32("llama.feed_forward_length", N_FF),
        kv_i32("llama.rope.dimension_count", N_ROT),
        kv_f32("llama.attention.layer_norm_rms_epsilon", RMS_EPS),
        kv_f32("llama.rope.freq_base", ROPE_THETA),
        kv_arr_str("tokenizer.ggml.tokens", VOCAB),
        kv_arr_f32("tokenizer.ggml.scores", [0.0] * N_VOCAB),
        kv_i32("tokenizer.ggml.bos_token_id", 1),
        kv_i32("tokenizer.ggml.eos_token_id", 2),
        kv_i32("tokenizer.ggml.unknown_token_id", 0),
        kv_str("tokenizer.ggml.model", "llama"),
    ]
    metadata = b"".join(kv)

    tensor_table = b""
    for name, dims, ttype, off in infos:
        tensor_table += gstr(name)
        tensor_table += struct.pack("<I", len(dims))
        for d in dims:
            tensor_table += struct.pack("<Q", d)
        tensor_table += struct.pack("<I", ttype)
        tensor_table += struct.pack("<Q", off)

    header = GGUF_MAGIC
    header += struct.pack("<I", VERSION)
    header += struct.pack("<Q", len(infos))
    header += struct.pack("<Q", len(kv))

    body = header + metadata + tensor_table
    pad = (-len(body)) % ALIGNMENT
    body += b"\x00" * pad
    body += b"".join(blobs)

    with open(args.out, "wb") as f:
        f.write(body)

    np.savez(args.out + ".npz", **effective)

    print(f"wrote {args.out} ({len(body)} bytes, {len(infos)} tensors, dtype={args.dtype})")
    print(f"wrote {args.out}.npz (reference weights)")


if __name__ == "__main__":
    main()
