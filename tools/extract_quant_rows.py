#!/usr/bin/env python3
"""Dump the first rows of real quantised tensors to raw blobs.

Big models are several gigabytes, which the engine's whole-file GGUF loader
will not swallow; this pulls just the bytes a kernel test needs. The blobs are
laid out row after row, exactly as they sit in the file, so the C side can read
them with the same row arithmetic it uses in the engine.

  extract_quant_rows.py <model.gguf> <type-id> [rows] [outdir]
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_probe import Reader  # noqa: E402

TYPE_NAMES = {12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 8: "Q8_0", 2: "Q4_0", 6: "Q5_0",
              23: "IQ4_XS", 39: "MXFP4", 10: "Q2_K", 11: "Q3_K",
              3: "Q4_1", 7: "Q5_1", 15: "Q8_K",
              41: "Q1_0", 42: "Q2_0", 142: "PQ2_0", 143: "PTQ1_0"}

# size of one block and elements per block, per ggml
SIZES = {2: (18, 32), 6: (22, 32), 8: (34, 32), 39: (17, 32), 10: (84, 256), 11: (110, 256),
         12: (144, 256), 13: (176, 256), 14: (210, 256), 23: (136, 256),
         3: (20, 32), 7: (24, 32), 15: (292, 256),
         41: (18, 128), 42: (18, 64), 142: (34, 128), 143: (28, 128)}


def main():
    path = sys.argv[1]
    want = int(sys.argv[2])
    rows = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    outdir = sys.argv[4] if len(sys.argv) > 4 else "build/kq"
    os.makedirs(outdir, exist_ok=True)

    with open(path, "rb") as f:
        r = Reader(f)
        magic = r.take(4)
        assert magic == b"GGUF", magic
        r.u32()                      # version
        n_tensors = r.u64()
        n_kv = r.u64()
        for _ in range(n_kv):
            r.string()
            t = r.u32()
            r.value(t)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            typ = r.u32()
            off = r.u64()
            tensors.append((name, dims, typ, off))
        align = 32
        data_off = (f.tell() + align - 1) // align * align

        written = 0
        for name, dims, typ, off in tensors:
            if typ != want or len(dims) < 2:
                continue
            k, n = dims[0], dims[1]
            bsz, bel = SIZES[typ]
            assert k % bel == 0, (name, k, bel)
            row_bytes = (k // bel) * bsz
            take = min(rows, n)
            f.seek(data_off + off)
            blob = f.read(row_bytes * take)
            if len(blob) != row_bytes * take:
                continue
            safe = name.replace("/", "_").replace(".", "_")
            out = os.path.join(outdir, "%s.bin" % safe)
            with open(out, "wb") as g:
                g.write(struct.pack("<IIII", typ, k, take, row_bytes))
                g.write(blob)
            print("%-44s %s  k=%-6d rows=%-3d row_bytes=%d"
                  % (name, TYPE_NAMES.get(typ, typ), k, take, row_bytes))
            written += 1
            if written >= 4:
                break
    print("wrote %d blob(s) to %s" % (written, outdir))


if __name__ == "__main__":
    main()
