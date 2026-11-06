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

The PrismML Bonsai family stores weights at 1.75–2.13 bits per weight. Three
packings are decoded here, per the layouts published in
`MODEL-FORMATS.md` / llama.cpp's `block_pq2_0` and `block_ptq1_0`:

| ggml type | id | group | bytes/block | bpw | layout |
|---|---|---|---|---|---|
| `Q1_0` | 41 | 128 | 18 | 1.125 | fp16 scale + 16 B of sign bits |
| `Q2_0` | 42 | 64 | 18 | 2.25 | fp16 scale + 2-bit slots |
| `PQ2_0` | 142 | 128 | 34 | 2.125 | fp16 scale + 2-bit slots |
| `PTQ1_0` | 143 | 128 | 28 | 1.75 | fp16 scale + 24 B of 5-trits-per-byte + 2 B of 4-trits-per-byte |

`Q1_0` has no zero state: weight `j` is `+d` when bit `j` of `qs` is set and `-d`
otherwise, with `d` the block's mean absolute weight. The scale comes first in
`Q1_0`/`Q2_0`/`PQ2_0` but last in `PTQ1_0`.

**This is storage support, not model support.** A Bonsai checkpoint still will
not run, and that is not a dequantisation problem:

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
--tokenize <text>   print token ids and the decode round-trip, then exit
                    ("-" reads the text from stdin, which is how non-ASCII
                    reaches the engine on Windows)
--version           print version and exit
--help              show usage
```

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
src/chat.c         conversation history and completion path
src/http.c         HTTP/1.1 server and OpenAI routes
src/main.c         CLI entry point
src/q4_matmul.c    quantised GEMV kernels
tools/             model generators, reference implementation, comparators
```

## License

MIT-style – free to use, modify and extend.
