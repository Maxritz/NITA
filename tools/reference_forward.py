#!/usr/bin/env python3
"""Independent numpy reference for NITA's llama forward pass.

Loads the weights written by ``tools/make_tiny_model.py`` and computes the
final logits for a token sequence, mirroring ``src/transformer.c``:

    embedding -> [RMSNorm, QKV projection, RoPE, causal attention with a KV
    cache, output projection, residual] x n_layer -> RMSNorm -> output
    projection.

Usage:
    python3 tools/reference_forward.py --npz build/tiny.gguf.npz --tokens 3,4,5
"""

import argparse

import numpy as np

N_EMBD = 128
N_LAYER = 2
N_HEAD = 4
N_HEAD_KV = 2
N_FF = 256
HEAD_DIM = N_EMBD // N_HEAD
KV_DIM = HEAD_DIM * N_HEAD_KV
N_ROT = HEAD_DIM
RMS_EPS = 1e-5
ROPE_THETA = 10000.0


def rmsnorm(x, w):
    ss = float(np.mean(np.asarray(x, dtype=np.float64) ** 2))
    return (x * np.float32(1.0 / np.sqrt(ss + RMS_EPS)) * w).astype(np.float32)


def rope(vec, n_heads, head_dim, n_rot, pos, theta):
    out = vec.astype(np.float32).copy()
    for h in range(n_heads):
        base = h * head_dim
        for i in range(n_rot // 2):
            freq = np.float32(theta) ** np.float32(-2.0 * i / n_rot)
            angle = np.float32(pos) * freq
            c = np.cos(angle)
            s = np.sin(angle)
            x0 = out[base + 2 * i]
            x1 = out[base + 2 * i + 1]
            out[base + 2 * i] = x0 * c - x1 * s
            out[base + 2 * i + 1] = x0 * s + x1 * c
    return out


def silu(x):
    return x / (np.float32(1.0) + np.exp(-x))


def forward(w, token_ids):
    n_vocab = w["token_embd.weight"].shape[0]
    k_cache = [np.zeros((0, KV_DIM), dtype=np.float32) for _ in range(N_LAYER)]
    v_cache = [np.zeros((0, KV_DIM), dtype=np.float32) for _ in range(N_LAYER)]
    logits = None

    for pos, tok in enumerate(token_ids):
        x = w["token_embd.weight"][tok].astype(np.float32).copy()

        for l in range(N_LAYER):
            h = rmsnorm(x, w[f"blk.{l}.attn_norm.weight"])

            q = (w[f"blk.{l}.attn_q.weight"] @ h).astype(np.float32)
            k = (w[f"blk.{l}.attn_k.weight"] @ h).astype(np.float32)
            v = (w[f"blk.{l}.attn_v.weight"] @ h).astype(np.float32)

            q = rope(q, N_HEAD, HEAD_DIM, N_ROT, pos, ROPE_THETA)
            k = rope(k, N_HEAD_KV, HEAD_DIM, N_ROT, pos, ROPE_THETA)

            k_cache[l] = np.vstack([k_cache[l], k[None, :]])
            v_cache[l] = np.vstack([v_cache[l], v[None, :]])

            attn = np.zeros(N_HEAD * HEAD_DIM, dtype=np.float32)
            inv = np.float32(1.0 / np.sqrt(HEAD_DIM))

            for hd in range(N_HEAD):
                kvh = hd * N_HEAD_KV // N_HEAD
                qh = q[hd * HEAD_DIM:(hd + 1) * HEAD_DIM]
                kh = k_cache[l][:, kvh * HEAD_DIM:(kvh + 1) * HEAD_DIM]
                vh = v_cache[l][:, kvh * HEAD_DIM:(kvh + 1) * HEAD_DIM]

                scores = (kh @ qh) * inv
                scores = scores - np.max(scores)
                probs = np.exp(scores)
                probs = probs / np.sum(probs)

                attn[hd * HEAD_DIM:(hd + 1) * HEAD_DIM] = probs @ vh

            x = x + (w[f"blk.{l}.attn_output.weight"] @ attn).astype(np.float32)

            h2 = rmsnorm(x, w[f"blk.{l}.ffn_norm.weight"])
            gate = (w[f"blk.{l}.ffn_gate.weight"] @ h2).astype(np.float32)
            up = (w[f"blk.{l}.ffn_up.weight"] @ h2).astype(np.float32)
            ff = silu(gate) * up
            x = x + (w[f"blk.{l}.ffn_down.weight"] @ ff).astype(np.float32)

        hout = rmsnorm(x, w["output_norm.weight"])
        logits = (w["output.weight"] @ hout).astype(np.float32)

    return logits, n_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tokens", required=True, help="comma separated token ids")
    args = ap.parse_args()

    data = np.load(args.npz)
    w = {k: data[k].astype(np.float32) for k in data.files}
    ids = [int(t) for t in args.tokens.split(",") if t.strip() != ""]

    logits, n_vocab = forward(w, ids)

    print("tokens: " + " ".join(str(i) for i in ids))
    print("logits: " + " ".join(f"{v:.6f}" for v in logits))
    _ = n_vocab


if __name__ == "__main__":
    main()
