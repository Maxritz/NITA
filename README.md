# NITA (NC Inference Tool for All)

CPU-only GGUF inference engine with an OpenAI-compatible HTTP API.
AVX2 target (AMD Zen3), quantised weights dequantised per row on demand.

## What works

| Area | Status |
|------|--------|
| GGUF v2/v3 parsing | **Working** – metadata (all 13 value types incl. arrays), tensor table, data offset |
| Weight dequantisation | **Working** – F32/F16/BF16/F64, I8/I16/I32, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K, Q1_0 (binary), Q2_0/PQ2_0/PTQ1_0 (ternary) |
| Tokenizer | **Working** – real byte-level BPE (`tokenizer.ggml.merges`), SentencePiece score merging, byte fallback |
| Transformer forward pass | **Working** – embeddings, RMSNorm, RoPE, grouped-query attention with KV cache, SwiGLU FFN, output projection |
| Sampling | **Working** – greedy, temperature, top-k, deterministic seed |
| OpenAI HTTP API | **Working** – `/health`, `/v1/models`, `/v1/chat/completions` (streaming and not) |
| Conversation history | **Working** – multi-turn `messages`, roles, usage accounting |

Generation is real model output. For a random-weight checkpoint that output is
nonsense words, which is expected; with a real llama-family GGUF it is decoded
model text.

### Ternary weights

The PrismML Bonsai family stores weights at 1.75–2.13 bits per weight. Four
packings are decoded here, per the block structs in the PrismML fork's
`ggml-common.h` (`block_q1_0`, `block_q2_0`, `block_pq2_0`, `block_ptq1_0`).
The two upstream ternary formats, `TQ1_0` and `TQ2_0`, use the same codings at
group 256 and are described under *Quantised-domain kernels*.

| ggml type | id | group | bytes/block | bpw | layout |
|---|---|---|---|---|---|
| `Q1_0` | 41 | 128 | 18 | 1.125 | fp16 scale + 16 B of sign bits |
| `Q2_0` | 42 | 64 | 18 | 2.25 | fp16 scale + 2-bit slots |
| `PQ2_0` | 142 | 128 | 34 | 2.125 | fp16 scale + 2-bit slots |
| `PTQ1_0` | 143 | 128 | 28 | 1.75 | fp16 scale + 24 B of 5-trits-per-byte + 2 B of 4-trits-per-byte |

Type id 42 has a history worth knowing before pointing NITA at a file. The
prism-v5 era stored the **group-128** layout under 42; mainline llama.cpp then
standardised 42 as **group 64**, and prism-v7 moved the fork's group-128 layout
to its own id, `PQ2_0` (142). The deprecated `*-Q2_0.gguf` files still published
on the HuggingFace repos are the old group-128 layout under 42, so every row of
them is 34 bytes per 128 weights where a current reader expects 18 per 64 - and
a reader that simply follows the mapping decodes them as garbage instead of
noticing.

NITA follows the current mapping and refuses those files. At load time each
type-42 tensor's span (the gap from its offset to the next tensor's, or to the
data section end for the last one) is compared with what group 64 needs: a span
too small for 18 bytes per 64 but large enough for 34 per 128 is the legacy
layout, and loading stops with

```
this file matches the legacy Prism Q2_0 layout (group size 128 stored as ggml
  type id 42): tensor 'blk.0.attn_q.weight' holds 68 bytes, but Q2_0 is read
  here as the official group-64 format and needs 72. Use the *_g64 or PQ2_0
  file for this model
```

The check is byte-span based, not name based, and only applies when the row
width is a multiple of 128 (the smallest width both layouts can describe). A
truncated tensor gets a separate message. The result is verified against
synthetic group-64, legacy group-128, truncated and `PQ2_0` files, and every
model in `build/` still loads.

`Q1_0` has no zero state: weight `j` is `+d` when bit `j` of `qs` is set and `-d`
otherwise, with `d` the block's mean absolute weight. The scale comes first in
`Q1_0`/`Q2_0`/`PQ2_0` but last in `PTQ1_0`.

### Hadamard rotation

Bonsai checkpoints store some matmul weights folded with a fixed rotation, so a
runtime has to rotate the *activation* before the matmul - the fold is not
decodable from the weights alone. `prism.hadamard.block_size` gives the block
width, `prism.hadamard.weight_names` the 150 folded tensors,
`prism.hadamard.inverse_weight_names` the ones folded the other way (the token
embedding), and `prism.hadamard.sign_values` the explicit +-1 sign vectors, one
per width in `prism.hadamard.sign_widths`.

`src/hadamard.c` implements it. The rotation the PrismML loader materialises is
`R[row][col] = (popcount(row & col) & 1 ? -1 : +1) / sqrt(block)`, applied as
`y = R (s * x)` for folded weights and `x = s * (R y)` for the inverse. That
matrix is the normalised Sylvester Walsh-Hadamard matrix, so the module runs the
radix-2 transform in place instead of building or multiplying an `n x n` matrix.
The three low strides (`1`, `2`, `4`), which cannot be vectorised as a
load-two-halves butterfly and would otherwise dominate the cost, are done
in-register on eight consecutive elements; everything above that is 8-lane
butterflies.

`tests/hadamard_test.c` checks the transform against the reference matrix built
directly from the parity definition (blocks 2 to 1024, max difference 0), against
orthogonality (`R R^T = I`, so `forward` twice is the identity and inner products
are preserved), the round trip through `nita_hadamard_inverse` with and without
sign vectors, and that the signs actually change the result. A 2048-element
forward (four 512-wide blocks) costs 975 ns on the AVX2 build and 5.5 us scalar;
the first version was 2160 ns until the in-register low strides. The C output
also matches a numpy `R @ x` to 3e-6 relative, which is float32 accumulation
against a float64 reference.

The module is not yet called from the model graph - `qwen35` still needs its
loader and per-block dispatch before a Bonsai checkpoint can run, so the
*storage support, not model support* note below still holds. All four decode into an integer
kernel as well, so a Bonsai-shaped model that did work would not pay the float
path for its ternary weights; see *Quantised-domain kernels*.

**Loading, dispatch and execution are in** for the hybrid architecture, on both
checkpoints. A Bonsai checkpoint loads: the loader detects `general.architecture = qwen35`, decides per block
whether it is a gated delta-net mixer or a dense-attention block (upstream's
rule: recurrent unless `(layer + 1) % full_attention_interval == 0`), binds the
mixer's own tensors - `attn_qkv`, `attn_gate`, `ssm_conv1d`, `ssm_alpha`,
`ssm_beta`, `ssm_dt.bias`, `ssm_a`, `ssm_norm`, `ssm_out` - and allocates the
per-layer recurrent state. `Ternary-Bonsai-2-2B-PTQ1_0.gguf` reports:

```
qwen35 hybrid   : 18 gated delta-net blocks, 6 dense-attention blocks (every 4-th), ssm 128 state x 16 heads, inner 2048
```

Both generations load - the 27B reports 48 gated delta-net blocks and 16
attention blocks - because the head width is taken from `attention.key_length`
when present instead of assuming `n_embd / n_head` (the 27B runs 24 heads of 256
over a 5120-wide embedding).

**The hybrid forward matches the reference runtime.** The 2B executes end to
end: the rotated embedding table is inverted after the lookup, each block
dispatches to the delta-net mixer or to gated attention (per-head q/k RMS norm,
rotary over 64 of 256 dims, sigmoid gate on the attention output), the shared
SwiGLU follows `post_attention_norm`, and the 150 folded matmuls get their
activation rotated first.

Against llama.cpp built from the PrismML fork - the runtime these checkpoints
were made for - on token `5,6,7`:

- All 248320 logits at cosine **0.999989**, same argmax, identical top-5
  (16, 15, 17, 18, 64); mean absolute difference 0.017, worst single logit 0.09.
- Every tagged tap across all 24 blocks agrees: `attn_norm`, the mixer's
  `conv_output_silu` / `q_conv_predelta` / `v_conv_predelta` / `z` /
  `final_output`, `attn_residual`, `ffn_out`, `l_out`, and `result_norm`.
  Cosines run 0.9998 to 1.00000; what is left is the difference between the
  integer kernels and ggml's own arithmetic.
- Greedy generation answers properly: "Count from 1 to 10" gives `1 2 3 ... 10`,
  "What is the capital of France?" gives `The capital of France is Paris.`

The 27B (`Bonsai-27B-Q1_0.gguf`, 64 blocks, 5120 wide, 48 value heads) runs too:
all 248320 logits at cosine **0.999961**, same argmax, mean absolute difference
0.022, worst logit 0.16, and 493 tagged taps across every block between 0.9988
and 1.00000. It needs 3.5 GiB resident and about 25 s per token single-threaded
here.

That model found a second convention error, invisible on the 2B because it has
as many value heads as key heads. A value head has to read key head
`h % n_k_heads`, not `h / (n_v_heads / n_k_heads)`: the fused GDN kernel indexes
its q/k rows as `iv1 % neq1`, and the unfused path gets the same mapping for
free from `ggml_repeat`, which tiles a tensor rather than splitting it into
contiguous runs. With 48 v heads over 16 k heads the blocked reading pairs each
value head with the wrong query, and the mixer output drops to cosine 0.81 while
every input to it stays exact - which is what localised it.

`tests/q35_heads_test.c` pins the convention, since no run of the 2B can. It
walks the 27B geometry (48 value heads over 16 key heads), checks that each key
head feeds an equal share, and reports both readings' tables; swapping the
mapping back to blocked indexing makes it fail in 76 places. It runs under CTest
with the rest of the suite (see *Tests* under Build).

Getting there took one real bug, found by measurement rather than inspection.
The delta-rule output must be scaled by `1/sqrt(head_v_dim)` before the gated
norm (the fused GDN kernel does exactly this), and this engine omitted it. That
looks like it should not matter, since an RMS norm divides out a constant
scale - but on the first tokens the mixer output is tiny, so the norm's epsilon
is comparable to mean(x^2) and the normalisation stops being scale-invariant.
The missing factor therefore propagated at nearly full strength: the mixer
output came out ~3x too large and 0.89 in direction before it, and exactly
right after. Two dead ends were ruled out first (reversing the conv tap order,
and blaming the gate projection `z`, which was in fact exact).

Two things are established rather than assumed. `ssm_a` holds `-exp(A_log)`
already, and exponentiating it again left the mixer's state undamped; fixing
that changed the output from noise to token-shaped text, so it was a real bug.
The rotary pairing is the rotate-half one `(i, i + n_rot/2)`: qwen35 reports
IMROPE, which ggml implements with half-split pairs, unlike the adjacent-pair
NORMAL mode the llama path uses. `NITA_Q35_TRACE=1` prints mean/max magnitude
after every block - useful because the activations stay bounded, which rules
out overflow and points at a convention error instead.

### The reference harness

The engine is now checked against the runtime the weights were made for, not
against guesses. `PrismML-Eng/llama.cpp` on the `prism` branch is cloned into
`build/prism-llama` and built with MinGW GCC (`cmake -B build-mingw -G Ninja
-DLLAMA_CURL=OFF -DGGML_OPENMP=OFF`), with one local patch: the
`THREAD_POWER_THROTTLING_STATE` block in `ggml/src/ggml-cpu/ggml-cpu.c` needs
the MSVC SDK headers, so it is compiled only `defined(_MSC_VER)`. Stock
llama.cpp cannot load these files at all - it rejects `token_embd.weight` as
"invalid ggml type 143".

`tools/llama_logits.c` links against that library and prints llama.cpp's logits
for an explicit token sequence, which is the same thing `nita --tokens` prints.
`NITA_FORK_DUMP=<path>` (fork side) and `NITA_Q35_DUMP=<path>` (engine side)
dump the tagged intermediates - `attn_norm-<l>`, `l_out-<l>`, and the mixer's
`conv_output_silu`, `q_conv_predelta`, `v_conv_predelta`, `z`, `final_output` -
so the two graphs can be compared tap by tap with `tools/compare_taps.py`. The
fork already tags its mixer internals; the only local addition is a
`cb(output, "gdn_output", il)` line before the gated norm, in
`src/models/qwen35.cpp`.

### Reading the dumps

`tools/compare_taps.py mine.txt fork.txt` prints the cosine and norm ratio of
every tap both sides wrote. Two things to know about the files: ggml renames
views to `<name> (view)`, and those lines are skipped (their second field is not
a count); and a tap can appear more than once, in which case the last line wins.

When a tap disagrees, check where the engine writes it before believing the
number. `q_conv_predelta` is the *normalised* q, so the engine dumps it after
`l2norm_heads`, not before - an earlier version dumped the raw conv output and
reported a cos of 0.87 for a query that was in fact correct.

Beyond that, a Bonsai checkpoint still will not run, and it is not a
dequantisation problem:

- Its `general.architecture` is `qwen35`, a hybrid attention model — roughly 75%
  linear (gated delta net) and 25% full attention. NITA implements dense
  llama-style attention and has no linear-attention path.
- Several matmul weights are stored in a Hadamard-rotated basis described by
  `prism.hadamard.*` metadata (`block_size`, ±1 `sign_values`, rotation and
  inverse weight lists). Unfolding that rotation into the activation is part of
  the model function; decoding the weights alone yields plausible-looking
  garbage.
- The MTP variants carry extra prediction heads this engine does not consume.

What does work today on such a file: `--inspect` reports its metadata and tensor
table, `--tokenize` uses its real 248k-piece vocabulary, and rows dequantise
correctly — verified against an independent Python encoder.

## Build

Requirements: CMake 3.20+, GCC/MinGW 13+ (tested with Strawberry GCC 13.2 on Windows).

```bash
mkdir -p build && cd build
cmake .. -DQUANTIZED_GEMV=ON -DGPTQ=ON
cmake --build .
```

Produces `nita` (plus a `minimal_engine` compatibility copy).

### CMake options

| Flag | Effect |
|------|--------|
| `-DQUANTIZED_GEMV=ON` | Required base flag |
| `-DTERNARY=ON` | Ternary weights {-1,0,1} |
| `-DGPTQ=ON` | 4-bit grouped GPTQ |
| `-DAWQ=ON` | Activation-aware weight quantisation |
| `-DBINARY_NET=ON` | BitNet 1-bit signs |
| `-DBF16=ON` / `-DFP16=ON` | Float matmul (scalar) |
| `-DMXFP8=ON` | Forces a compile-time error (not on AVX2) |
| `-DSHERLOCK_LITE=ON` | Component trace table on stdout |
| `-DDEBUG_BREADCRUMBS=ON` | Breadcrumb markers on stderr |

Exactly one quantisation flag should be enabled; the build derives
`QUANT_TYPE` from it. The transformer itself reads the per-tensor types
recorded in the GGUF file, so a single binary handles every quantised tensor
found in a checkpoint.

### Tests

```bash
cmake --build build && ctest --test-dir build --output-on-failure
```

Everything but `main()` is built into a `nita_core` library that the tests link,
so a test picks up the same sources and flags as the engine.

| Test | Target | Needs a checkpoint |
|------|--------|--------------------|
| `hadamard` | `hadamard_test` | no |
| `q35_heads` | `q35_heads_test` | no |
| `q35_probe` | `q35_probe` | yes |
| — | `quant_dot_test` | yes, plus a type id: `<model.gguf> <type-id> [rows]` |

The two that read a GGUF register only when one is named:

```bash
cmake -S . -B build -DNITA_TEST_MODEL=/path/to/model.gguf
```

`quant_dot_test` stays a plain target because it also takes a tensor type: it
compares one type's kernel against the float reference over the tensors a
checkpoint actually stores, as `build/quant_dot_test <model.gguf> <type-id>`.

## Run

```bash
./nita --model path/to/model.gguf --port 7777
```

```
--model <path>      GGUF model to load (required)
--host <addr>       bind address (default 0.0.0.0)
--port <n>          listen port (default 7777)
--ctx <n>           context / KV-cache size (default 2048)
--name <id>         model id served by /v1/models (default: file stem)
--quant-type <name> label reported for the compiled kernel
--inspect           print model metadata and tensor table, then exit
--logits <text>     tokenize text, run the forward pass, print logits
--tokens <a,b,c>    run the forward pass on explicit token ids
--prefill <n>       measure prompt processing: n tokens evaluated in one
                    pass (NITA_PREFILL_SEQ=1 for the old one-at-a-time path)
--bench <n>         measure decode throughput and print the component profile
--reps <n>          runs per benchmark, reported as a median with a spread
                    warning (default 5; 1 = the old single-run behaviour).
                    Each run also reports its own CPU load, see below
--tokenize <text>   print token ids and the decode round-trip, then exit
                    ("-" reads the text from stdin, which is how non-ASCII
                    reaches the engine on Windows)
--version           print version and exit
--help              show usage
```

Environment knobs, all optional:

```
NITA_THREADS=<n>    worker threads (default: half the logical processors)
NITA_SPIN=<n>       spin iterations before a worker blocks (1 = never block)
NITA_TILE=<n>       prefill tile width, 1..64 (default 8)
NITA_PREFILL_SEQ=1  evaluate prompt tokens one at a time (A/B baseline)
NITA_ATTN_SEQ=1     keep tile attention on the calling thread (A/B baseline)
NITA_REP_SPREAD=<p> spread in percent that triggers the benchmark warning
                    (default 10, 0 disables)
NITA_REP_LOAD=<n>   external cores above which a run is left out of the
                    median; 0 or unset derives the limit from the quietest
                    run (see below)
NITA_INT8_DOT=0     use the old dequantise-to-float dot (A/B on one binary)
```

### Quantised-domain kernels

Weights are never expanded to fp32. Each kernel consumes the quantised block
format, multiplies in integer registers, and applies the block scale to the
int32 accumulator - one float multiply-add per block instead of per element:

```
for each 32-element block:
    w8   = expand this block of weights to int8   (vector, type specific)
    s    = sum_i xq_i * w8_i                     (int32, no floats)
    acc += (d_w * d_x) * s                       (one FMA)
```

Activations are quantised to int8 once per row (Q8_0 style) and reused by
every weight row. Types with a per-block minimum (Q4_K, Q5_K) subtract it as a
second term from the sum of the activations.

Covered: **Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
Q8_K, IQ4_XS, MXFP4, Q1_0, Q2_0, PQ2_0, PTQ1_0, TQ1_0, TQ2_0** - every ggml
quantised format the loader decodes now has an integer kernel, so no decoded
weight reaches the float path (`build/chk_types.exe` prints the list).

The two 4/5-bit legacy formats fold their offset into the int8 weight, so they
cost no more than a linear dot: Q4_0 subtracts 8 from each nibble
(`nib - 8`), Q5_0 ORs 0xF0 into the nibble when the 5th bit is clear, which is
`nib - 16` in two's complement. Neither needs a second accumulator. The K-quants
Q2_K and Q3_K instead carry a scale per *16* elements, so they keep a 4-lane
accumulator per half-block; Q2_K also has a per-half minimum, which is a second
dot against a vector of ones over the same activations.

Q4_1 and Q5_1 are the same shapes with a per-block float minimum instead of an
integer offset, so their minimum cannot be folded into the weight: it becomes
a second term driven by the sum of the activations, exactly as in Q4_K. Q8_1
and Q8_K store their weights as int8 already and are a plain signed dot with
one scale per block (Q8_1) or per 256 (Q8_K).

The ternary family (Q1_0, Q2_0, PQ2_0, PTQ1_0) has one scale per block and no
minimum, so the dot needs no correction term: all the work is turning packed
signs or 2-bit slots into `+-1`. Packed signs expand by replicating each stored
byte eight times and shifting it down, and 2-bit slots by replicating each byte
four times and masking after a 2-, 4- and 6-bit shift, with
`_mm256_blendv_epi8` dropping each slot at its own element. That is one load,
one shuffle and a few shifts per 32 weights, which is cheaper than the eight
table loads it replaced and - the reason it was replaced - contains no
data-dependent address: gcc -O2 fused those loads into a gather, and the gather
faulted about one run in five (`q2_expand` segfaulted at -O2, while -O1, gdb
and every ASLR-disabled run were clean). PTQ1_0 carries the same trit coding at group 128.
Five base-3 trits share a stored byte, the trit of byte `b` at plane `n` being
`floor(b * 3^(n+1) / 256) mod 3` - a remainder chain (`t = 3*r`, trit = `t >>
8`, `r = t & 0xFF`) rather than a table, three ops per plane for the whole lane
batch. At 24 stored bytes the planes come out 16 and 8 values wide instead of
32, so the four chunks are assembled by pairing planes with `_mm_unpacklo_epi64`
and `_mm256_set_m128i`; that is the only difference from the wider formats, and
nothing round-trips through memory.

Two upstream ternary formats, `TQ1_0` (34) and `TQ2_0` (35), use the same
codings at group 256. `TQ2_0` is 64 stored bytes with the fp16 scale *after*
them, and its 2-bit slots are bit planes: slot `l` covers 32 consecutive
elements rather than interleaving a slot per byte as `Q2_0`/`PQ2_0` do. That
makes it the cheapest ternary kernel here - each 32-element chunk is one load,
one shift, one mask, one subtract and one dot, with no blending. `TQ1_0` is
`PTQ1_0`'s trit packing widened to 48 bytes (`qs[48]`, `qh[4]`, scale last); at
that width the first five trit planes are exactly the first five 32-element
chunks, so those dots run straight from registers, and only the last three
chunks are assembled from the 16-wide planes plus the `qh` tail.

IQ4_XS and MXFP4 are codebook types: the 4-bit field is an *index*, not a
linear value, so the block expands through a 16-entry table before the integer
dot. The tables are the ggml ones, imported verbatim (`src/quant.c`, mirrored
in `src/dequant.c` so the reference path and the kernel cannot drift).

| type | block | bytes | block scale | value of code `i` |
|---|---|---|---|---|
| IQ4_XS | 256 | 136 | fp16 `d`, per-32 sub-block `d*(ls-32)`, `ls` = 4 bits of `scales_l` + 2 bits of `scales_h` | `{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}[i]` |
| MXFP4 | 32 | 17 | E8M0 byte `e` -> `2^(e-128)` | `{0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}[i]` (e2m1 doubled; the scale carries the missing half) |

For both, the 16 code bytes of a 32-element block hold elements 0..15 in the
low nibbles and 16..31 in the high nibbles, so one `_mm256_shuffle_epi8`
against the broadcast table turns the nibbles into int8 weights and the dot
runs exactly like the linear types.

`tests/quant_dot_test.c` checks each kernel against the float reference on real
tensor bytes (~1e-4 relative, the activation-quantisation floor). A kernel that
agrees with the reference is not yet correct, since both can misread the same
format, so each new type is also decoded in Python and compared against the
original f32 tensor it was quantised from. IQ4_XS was
also checked against `Laguna-XS.2-IQ4_XS.gguf` (16.84 GiB, 678 tensors, type 23
throughout) and MXFP4 against a tensor quantised by `llama-quantize`: decoded
against the original f32 weights, IQ4_XS correlates at 0.997 and MXFP4 at 0.994,
with the residual error at each format's step size rather than a layout mismatch.
End to end, a model with either tensor type produces the same argmax and 0.998/
0.989 logit cosine as its f32 original, and the int8 path agrees with
dequantise-then-float at 0.9998 cosine inside the engine.

The same check on the newer types (each quantised by `llama-quantize` from a
known f32 tensor) shows correlation 0.996 for Q4_0, 0.997 for Q4_1, 0.999 for
Q5_0, 0.999 for Q5_1, 0.950 for Q2_K and 0.987 for Q3_K, with the residual at
each format's step size - 2 bits can only reach about 0.95. All keep the argmax
of the f32 model and agree with the float path at 0.9998 cosine.

Q8_1 and Q8_K are dot intermediates that ggml refuses to write as tensor types
(its quantizer trips an assertion), so there is no ggml-produced file to check
them against. Their test blocks are built from a known f32 tensor with the
quantisers transcribed from `ggml-quants.c`, and decode back at 0.39% of the
block maximum - exactly the 1/127 step those formats promise.

`TQ1_0` and `TQ2_0` get the same treatment, since no local checkpoint carries
them: their test blocks are packed with upstream's `quantize_row_tq1_0_ref` and
`quantize_row_tq2_0_ref` loops transcribed verbatim, so decoding a block and
comparing it with the weights that went in checks element order against the
reference rather than against NITA's own decoder. Every one of the 256 weights
in every test block comes back exactly (scalar and AVX2 alike), and the kernels
agree with dequantise-then-float at 7.1e-4 and 6.3e-4. In the engine,
`tiny_tq1`/`tiny_tq2` sit 0.30 from the f32 original in the worst logit - the
same activation-quantisation band as the other ternary models, whose int8 paths
measure 0.32 and 0.53 - while the int8 path agrees with the float path at
0.99997 cosine with the same argmax. Those two models only quantise `ffn_down`,
the one tensor whose row width is a multiple of 256; everything else stays f32,
so the engine test exercises the kernel without running a fully ternary model.

That check paid for itself on Q3_K: the scale unpack in `dequant.c` rotated the
four scale words, so every sub-block was multiplied by a neighbouring block's
scale. The decoded tensor correlated at 0.33 with the original - and the int8
kernel, written against the same table, happily agreed with it. Fixed in both
places; this was a live correctness bug in the float path for any Q3_K model,
not just a kernel problem.

Q8_1 caught a second one. Its quantiser uses a `-128/max` scale where Q8_0 uses
`-127/max`, so a stored byte may legitimately be `-128` - and neither fast int8
dot can represent that value. The sign trick takes `-(-128)`, which saturates
back to `-128`; biasing the weight into 0..255 instead saturates the int16
`maddubs` intermediate at 255*127. Both cost about 3-9% relative error on real
blocks, which is the kind of wrong that still looks plausible. Q8_1 therefore
takes an exact widening path (int8 -> int16 -> `madd_epi16`, where every product
fits), at a small cost that shows up in its row above.

Prefill, Llama-3.2-1B Q8_0, 256 tokens, tile 16: 6.60 -> 5.34 ms/token. Decode
is memory bound and barely moves (43 -> 42 ms/token). Logits against the exact
path: same argmax, identical top-10, max |delta| 0.27, cosine 0.99982.

#### What a codebook dot costs

`quant_dot_test --bench <blob> [iters]` times one row through the integer
kernel and through dequantise-then-float, on the same bytes and the same
activations. Rows are real tensor rows pulled from the models named, k is the
row width, and `ns/row` is what the per-token decode loop pays for one weight
row. Median of three runs, 2000 rows each:

| type | source | k | int8 ns/row | ps/weight | dequant+float | speedup |
|---|---|---|---|---|---|---|
| Q8_K | synthesised, ggml's Q8_K quantiser | 2048 | 72.4 | 35.4 | 2481.4 | x34.3 |
| TQ2_0 | synthesised, upstream TQ packer | 2048 | 83.1 | 40.6 | 2808.9 | x33.8 |
| IQ4_XS | Laguna-XS.2-IQ4_XS | 2048 | 121.6 | 59.4 | 2643.9 | x21.8 |
| Q1_0 | synthesised ternary blocks | 2048 | 130.1 | 63.5 | 3165.5 | x24.3 |
| PQ2_0 | synthesised ternary blocks | 2048 | 138.2 | 67.5 | 3234.8 | x23.4 |
| Q2_0 | synthesised ternary blocks | 2048 | 167.2 | 81.6 | 3341.1 | x20.0 |
| PTQ1_0 | synthesised ternary blocks | 2048 | 182.6 | 89.2 | 3008.3 | x16.5 |
| Q8_1 | synthesised, ggml's Q8_1 quantiser | 2048 | 194.9 | 95.2 | 2483.0 | x12.7 |
| Q4_0 | tiny, via `llama-quantize` | 2048 | 195.3 | 95.4 | 2500.2 | x12.8 |
| MXFP4 | tiny, via `llama-quantize` | 2048 | 205.6 | 100.4 | 4341.1 | x21.1 |
| Q5_0 | tiny, via `llama-quantize` | 2048 | 207.3 | 101.2 | 3065.3 | x14.8 |
| Q4_K | Llama-3.2-1B blend | 3840 | 391.3 | 101.9 | 5413.7 | x13.8 |
| Q4_1 | tiny, via `llama-quantize` | 2048 | 228.5 | 111.6 | 2566.8 | x11.2 |
| Q2_K | tiny, via `llama-quantize` | 2048 | 229.9 | 112.2 | 2730.7 | x11.9 |
| Q5_1 | tiny, via `llama-quantize` | 2048 | 233.8 | 114.2 | 3022.1 | x12.9 |
| TQ1_0 | synthesised, upstream TQ packer | 2048 | 248.9 | 121.5 | 2942.1 | x11.8 |
| Q6_K | GLM-4.7-Flash-APEX | 12288 | 1600.9 | 130.3 | 21195.6 | x13.2 |
| Q3_K | tiny, via `llama-quantize` | 2048 | 281.3 | 137.4 | 3237.6 | x11.5 |
| Q5_K | GLM-4.7-Flash-APEX | 4096 | 647.8 | 158.2 | 6660.3 | x10.3 |

Q1_0 runs at 63.5 ps/element, behind only TQ2_0 and IQ4_XS and a third under
Q4_0's 95.4: 128 weights share one fp16 scale, and a 32-weight dot is a load, a
shuffle and a few shifts with no minimum term and no second accumulator. PQ2_0 matches it (67.5 ps) on the same 128-weight
block, and Q2_0 pays 14 ps more only because its blocks are 64 weights wide, so
every scale load, multiply and FMA lands twice as often.

TQ2_0 is the cheapest ternary kernel at 40.6 ps/element, behind only Q8_K,
because its slots are bit planes: one 32-element chunk is a coalesced load, a
shift, a mask, a subtract and a dot, with no interleaving to undo. TQ1_0 pays
121.5 ps - five remainder steps per 16 lanes - and though its first five planes
dot straight from registers, its last three chunks have to be assembled from
16-wide halves, which lands it between the linear 4-bit types and Q6_K rather
than beside the other ternary formats.

PTQ1_0 was the outlier here for a while: a table lookup per weight plus a
128-byte staging buffer per block put it at 414 ps/element, six times Q1_0 for
the same 128-weight block. Expanded with the same remainder chain as TQ1_0 it
lands at 89.2 ps, between Q2_0 and Q8_1, so the buffer and the per-weight
address were the whole cost rather than the trit packing itself. Its trits do
not sit on element boundaries - one stored byte feeds five different elements -
but that only changes how the planes are paired into chunks, not the arithmetic
per plane.

Q8_K is the cheapest weight of all at 35 ps/element and 32 GB/s: it is already
int8, so a block is one 32-byte load and one `dpbusd` with a single scale per
256 weights, which is why it exists as a dot intermediate in the first place.
Q4_1 and Q5_1 pay about 15 ps more than their integer-offset twins for the
per-block float minimum and its extra term.

Per weight Q2_K is dearer than Q4_0 even though it stores half the bits: its
minimum is per 16 elements, so it pays a second dot against a ones-vector and a
second FMA for every half-block, where Q4_0's offset is free inside the int8
weight. That is a cost paid per weight row, not per token - the sums of the
activations it needs are the same for every row, so precomputing them once per
projection would remove most of the gap. Q3_K sits between them because it
keeps a scale per 16 elements and no minimum.

IQ4_XS is the cheapest weight in the set at 60.6 ps/element, well under Q4_K's
101.9 despite carrying the same 4 bits plus a codebook: 136 bytes cover 256
weights, its 6-bit scales unpack with two shifts, and a whole 32-weight dot is
one shuffle plus one `dpbusd`. MXFP4 costs about what Q4_K does, because its
blocks are 8x shorter - a scale load, an E8M0 table lookup and a broadcast land
on every 32 weights instead of every 256 - which the 17-byte block size does not
pay back. A 4-bit codebook dot is therefore not inherently dearer than a linear
4-bit one; the only thing the codebook adds is the byte shuffle against the 16
entries, and it removes the min/subtract term that Q4_K needs.

The MXFP4 dequantise-then-float reference is slower than the other types' at
the same k because that path calls `exp2f` once per 32-element block, which is
exactly what the kernel avoids by using a 256-entry table.

### Why a run was slow

A median hides the reason one run drifted, so every run in a repeated
benchmark records the load it ran under:

```
  load: cpu = our cores busy, sys = whole machine busy, other = the rest
  run 1: 0.872 s  6.81 ms/token   cpu 11.2  sys 51%  other 5.2
  run 2: 0.833 s  6.51 ms/token   cpu 10.5  sys 42%  other 3.0

prefill: 128 tokens in 0.869 s -> 147.3 tok/s (6.79 ms/token) [tiled] (median)
  spread: 4.5% over 3 runs (min 6.51, max 6.81 ms/token)
  load: fastest run cpu 10.5 sys 42% other 3.0 | slowest run cpu 11.2 sys 51% other 5.2 (32 CPUs)
```

`cpu` is how many cores this process kept busy, `sys` is the fraction of the
whole machine that was busy, and `other` is the difference in cores, i.e. the
load competing with the benchmark. A run that was slow under a high `other`
was waiting on the rest of the machine; one that was slow with `other` flat
was slow for its own reasons. The counters come from `GetProcessTimes` and
`GetSystemTimes` bracketing each run, so they cost nothing measurable and do
not touch the component profile.

### Runs from a polluted machine are not counted

A run that took twice as long because something else grabbed the machine is
not evidence about this engine, so runs above the load limit are left out of
the median, the spread and the component table:

```
  run 2: 0.873 s  6.82 ms/token   cpu 11.3  sys 43%  other 2.5
  run 3: 0.989 s  7.73 ms/token   cpu 11.2  sys 73%  other 12.2   [excluded: load]

prefill: 128 tokens in 0.857 s -> 149.4 tok/s (6.70 ms/token) [tiled] (median)
  excluded: 6 of 8 runs, external load above 4.5 cores (worst run 3 at 12.2)
```

The limit is `quietest run + max(half of it, 2 cores)`, so it follows the
machine rather than a fixed number: when the whole session is busy, nothing
looks unusual and nothing is excluded; when load arrives halfway through,
only the polluted runs go. The decision uses the load counters and never the timings, so it
cannot select the answer it likes; two runs always survive, and when fewer
than three do the report says the median is really a mean. `NITA_REP_LOAD`
pins the limit when you want one. On a machine without the system counters
there is nothing to compare and no run is ever excluded.

## API

```bash
curl http://localhost:7777/health
curl http://localhost:7777/v1/models

curl -X POST http://localhost:7777/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nita","messages":[{"role":"user","content":"Hello"}],
       "max_tokens":64,"temperature":0.8,"top_k":40}'
```

Add `"stream": true` for server-sent events in the OpenAI chunk format,
terminated by `data: [DONE]`.

`system_fingerprint` reports which path served the request:

- `fp_nita` – real generation through the transformer
- `fp_nita_stub` – the checkpoint has no usable transformer weights, so a
  labelled acknowledgement was returned instead

## Verifying the forward pass

`tools/make_tiny_model.py` writes a complete, randomly-initialised llama-style
GGUF plus a `.npz` of the same weights. `tools/reference_forward.py` is an
independent numpy implementation of the same maths, and
`tools/compare_logits.py` runs both and reports the maximum logit difference:

```bash
python3 tools/make_tiny_model.py build/tiny.gguf --dtype f32
python3 tools/compare_logits.py --gguf build/tiny.gguf --npz build/tiny.gguf.npz --tokens 3,4,5,6
```

The engine matches the reference to about 2e-6 across sequence lengths 1–8,
for F32 and Q8_0 weights, and to 1.4e-5 for ternary weights — the larger figure
is fp32 accumulation over ±1 weights, not a decode error. `--dtype` accepts
`f32`, `q8_0`, `q2_0`, `pq2_0` and `ptq1_0`. This checks the whole path:
embedding lookup, RMSNorm, RoPE, grouped-query attention with the KV cache,
SwiGLU, output projection and block dequantisation.

Ternary and binary codecs are additionally checked for exactness: the encoders
are lossless for ±1 weights, so a transposed or mispacked trit shows up as a
hard failure rather than as plausible logits.

`tools/gguf_probe.py` validates a type's block geometry against a real file: it
reads only the header and tensor table, then checks that every tensor's declared
size adds up to the file's data section. On `Bonsai-27B-Q1_0.gguf` the `Q1_0`
entry accounts for 3,781,877,760 B of a 3,792,459,776 B data section — a +0 B
difference, with the last tensor ending exactly at EOF, which pins 128 elements
per 18-byte block (1.125 bpw). An off-by-one block size would misplace every one
of the 851 tensor offsets.

## Tokenizer

The encoder is selected from the model's own metadata:

| `tokenizer.ggml.model` | Encoder |
|---|---|
| `gpt2` | Byte-level BPE over the GPT-2 byte↔unicode table, merging by `tokenizer.ggml.merges` rank |
| `llama`, `spm` | SentencePiece: spaces become U+2581, adjacent symbols merge by token score |
| absent, but merges present | BPE |
| anything else | Longest-match over the vocabulary |

Bytes with no vocabulary entry fall back to `<0xXX>` tokens (SentencePiece) or
the single-byte token (GPT-2). Decoding reverses the byte mapping, turns `<0xXX>`
back into bytes, and drops the one leading space SentencePiece prefixes.

```bash
./nita --model model.gguf --tokenize "hello world"
printf 'h\xc3\xa9llo' | ./nita --model model.gguf --tokenize -
```

`tools/reference_tokenizer.py` is an independent Python implementation of both
encoders and compares token ids against the engine on a generated vocabulary.
Verified there on ASCII, unicode, whitespace and byte-fallback cases, and against
a real 248k-piece Qwen-family vocabulary (round-trip exact, and common
punctuation lands on its canonical ids: `,`→11, `!`→0).

## Performance and memory

The forward pass is deliberately simple: scalar dot products over rows that are
dequantised into a reusable buffer. It is correct but slow — expect well under
one token per second on a multi-billion-parameter model. The KV cache is
`2 x n_layer x ctx x (n_head_kv * head_dim) x 4` bytes, so a 32-layer model
with 4096-wide KV at 2048 context needs roughly 2 GB of cache.

Fused quantised GEMV kernels (avoiding the per-row expansion entirely) are the
natural next optimisation, along with multi-threading over output rows.

## Layout

```
src/nita.h          shared types, tracing macros, module APIs
src/util.c         growable buffers, allocation helpers
src/gguf.c         GGUF v2/v3 parser
src/dequant.c      block dequantisation for every supported tensor type
src/json.c         JSON parser and escaper
src/tokenizer.c    vocabulary + encode/decode
src/transformer.c  tensor binding, forward pass, sampling, generation
src/qwen35.c       gated delta-net mixer for the Bonsai hybrid architecture
src/quant.c        IQ4_XS / MXFP4 / TQ1_0 codecs and integer dot kernels
src/hadamard.c     block Hadamard rotation folded into the Bonsai matmuls
src/chat.c         conversation history and completion path
src/http.c         HTTP/1.1 server and OpenAI routes
src/main.c         CLI entry point
src/q4_matmul.c    quantised GEMV kernels
tests/             CTest entry points (see Tests under Build)
tools/             model generators, reference implementation, comparators
```

## License

MIT-style – free to use, modify and extend.
