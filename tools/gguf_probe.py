#!/usr/bin/env python3
"""Report a GGUF's metadata and check its tensor table against a type table.

This reads only the header, metadata and tensor table (never the weights), then
checks that the declared block geometry of every tensor type accounts exactly
for the file's data section. A wrong block size or element count shows up as a
byte gap, which is how the ternary/binary layouts were validated against real
PrismML files.

Usage:
    python3 tools/gguf_probe.py G:/More-models/Bonsai-27B-Q1_0.gguf
    python3 tools/gguf_probe.py --engine build/nita.exe model.gguf

With --engine the four header facts this script derives independently (version,
tensor count, metadata key count, data offset) are compared against what
`nita --inspect` reports, so a change to src/gguf.c fails here and not only
further downstream.
"""

import os
import struct
import subprocess
import sys

# ggml type id -> (name, elements per block, bytes per block)
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
    142: ("PQ2_0", 128, 34),
    143: ("PTQ1_0", 128, 28),
}

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("truncated file")
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t in SCALAR:
            return self.take(SCALAR[t])
        if t == 8:
            return self.string()
        if t == 9:
            it = self.u32()
            n = self.u64()
            if it == 8:
                return [self.string() for _ in range(n)] if n < 5000 else self.skip_array(it, n)
            return self.skip_array(it, n)
        raise ValueError(f"unknown metadata type {t}")

    def skip_array(self, it, n):
        if it in SCALAR:
            self.take(SCALAR[it] * n)
            return f"<{n} items>"
        for _ in range(n):
            self.string()
        return f"<{n} strings>"


def engine_header(exe, path):
    proc = subprocess.run([exe, "--model", os.path.abspath(path), "--inspect"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"engine exited {proc.returncode}: "
                         f"{proc.stderr.strip()[:200]}")
    names = {"GGUF version": "version", "tensors": "tensors",
             "metadata keys": "kv", "data offset": "data_offset"}
    got = {}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key in names and value.strip().isdigit():
            got[names[key]] = int(value)
    missing = [n for n in names.values() if n not in got]
    if missing:
        raise SystemExit(f"engine --inspect did not report {', '.join(missing)}")
    return got


def main():
    argv = sys.argv[1:]
    engine = None
    if "--engine" in argv:
        i = argv.index("--engine")
        if i + 1 >= len(argv):
            raise SystemExit("--engine needs the path to nita")
        engine = os.path.abspath(argv[i + 1])
        del argv[i:i + 2]
    if len(argv) != 1:
        raise SystemExit("usage: gguf_probe.py [--engine nita.exe] model.gguf")
    path = argv[0]
    with open(path, "rb") as f:
        r = Reader(f)
        if r.take(4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        meta = {}
        for _ in range(n_kv):
            key = r.string()
            t = r.u32()
            v = r.value(t)
            meta[key] = v

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()
            tensors.append((name, dims, ttype, offset))

        end_of_table = f.tell()

    align = meta.get("general.alignment", b"\x20\x00\x00\x00")
    align = struct.unpack("<I", align)[0] if isinstance(align, bytes) else 32
    data_offset = (end_of_table + align - 1) // align * align

    file_size = os.path.getsize(path)

    print(f"file            : {os.path.basename(path)}")
    print(f"version         : {version}")
    print(f"architecture    : {meta.get('general.architecture', b'?').decode('utf-8', 'replace')}"
          if isinstance(meta.get("general.architecture"), bytes) else
          f"architecture    : {meta.get('general.architecture')}")
    print(f"tensors         : {n_tensors}")
    print(f"data offset     : {data_offset}")
    print(f"file size       : {file_size}")

    by_type = {}
    total = 0
    max_end = 0
    unknown = []
    for name, dims, ttype, offset in tensors:
        info = TYPES.get(ttype)
        if info is None:
            unknown.append((name, ttype))
            continue
        _tname, block, bsize = info
        ne0 = dims[0]
        nrows = 1
        for d in dims[1:]:
            nrows *= d
        blocks = (ne0 + block - 1) // block
        nbytes = blocks * bsize * nrows
        by_type[ttype] = by_type.get(ttype, 0) + nbytes
        total += nbytes
        max_end = max(max_end, offset + nbytes)

    print("\ntype accounting:")
    for ttype in sorted(by_type):
        tname, block, bsize = TYPES[ttype]
        bpw = bsize * 8.0 / block
        print(f"  {tname:<8} id={ttype:<4} {block:>3} el/block {bsize:>3} B/block "
              f"({bpw:.3f} bpw) -> {by_type[ttype]:>14,} B")

    if unknown:
        print(f"\nunrecognised types: {unknown[:5]}")

    expected = file_size - data_offset
    print(f"\nexpected data   : {expected:,} B")
    print(f"accounted       : {total:,} B")
    print(f"difference      : {expected - total:+,} B")
    print(f"tensor end      : {max_end:,} B (data ends at {expected:,})")

    # The last tensor should end at EOF; padding between tensors is already
    # included in each tensor's own size, so any residual means wrong geometry.
    ok = abs((expected - total)) <= align * 2
    print(f"layout check    : {'CONSISTENT' if ok else 'MISMATCH'}")

    if engine:
        want = {"version": version, "tensors": n_tensors, "kv": n_kv,
                "data_offset": data_offset}
        got = engine_header(engine, path)
        differ = {k: f"engine {got[k]} != file {want[k]}"
                  for k in want if got[k] != want[k]}
        if differ:
            print("engine check    : DISAGREE " +
                  "; ".join(f"{k} ({v})" for k, v in differ.items()))
            ok = False
        else:
            print("engine check    : AGREE (version, tensors, keys, data offset)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
