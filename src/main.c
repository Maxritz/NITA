/*
 * main.c – NITA command line entry point.
 *
 *   nita --model <path.gguf> [--port 7777] [--host 0.0.0.0]
 *        [--ctx 2048] [--name <served-model-id>] [--quant-type Q]
 *        [--inspect] [--version] [--help]
 *
 * --inspect prints the GGUF metadata and tensor table, then exits.
 */

#include "nita.h"

#include <stdlib.h>
#include <string.h>

/* Kernel entry point provided by q4_matmul.c (one implementation is
 * compiled in, chosen by the CMake -D flags). */
extern void q4_matmul_kernel(const float *restrict scales,
                             const uint8_t *restrict A,
                             const float *restrict B_scales,
                             const int32_t *restrict B,
                             int32_t *restrict out,
                             int M, int N, int K);

/* Quantisation type the binary was built with.  The CMake flags define each
 * selector as the integer 1, so the human-readable name must be chosen with
 * an explicit #if chain rather than by stringizing QUANT_TYPE. */
#ifndef QUANT_TYPE
#  define QUANT_TYPE Q4_K_M
#endif

#if defined(TERNARY) && QUANT_TYPE == TERNARY
#  define NITA_QUANT_NAME "TERNARY"
#elif defined(GPTQ) && QUANT_TYPE == GPTQ
#  define NITA_QUANT_NAME "GPTQ"
#elif defined(AWQ) && QUANT_TYPE == AWQ
#  define NITA_QUANT_NAME "AWQ"
#elif defined(BINARY_NET) && QUANT_TYPE == BINARY_NET
#  define NITA_QUANT_NAME "BINARY_NET"
#elif defined(BF16) && QUANT_TYPE == BF16
#  define NITA_QUANT_NAME "BF16"
#elif defined(FP16) && QUANT_TYPE == FP16
#  define NITA_QUANT_NAME "FP16"
#elif defined(MXFP8) && QUANT_TYPE == MXFP8
#  define NITA_QUANT_NAME "MXFP8"
#else
#  define NITA_QUANT_NAME "Q4_K_M"
#endif

static void print_usage(FILE *f, const char *argv0)
{
    fprintf(f,
        "NITA %s - NC Inference Tool for All\n"
        "\n"
        "Usage:\n"
        "  %s --model <path.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --model <path>      GGUF model to load (required)\n"
        "  --host <addr>       bind address (default 0.0.0.0)\n"
        "  --port <n>          listen port (default 7777)\n"
        "  --ctx <n>           context size reported to clients (default 2048)\n"
        "  --name <id>         model id served by /v1/models (default: file stem)\n"
        "  --quant-type <name> label to report for the compiled kernel\n"
        "  --inspect           print model metadata and exit\n"
        "  --tokenize <text>   print token ids and decode round-trip, then exit\n"
        "                      (\"-\" reads the text from stdin)\n"
        "  --logits <text>     run the forward pass on text, print logits\n"
        "  --tokens <a,b,c>    run the forward pass on explicit token ids\n"
        "  --version           print version and exit\n"
        "  --help              show this message\n"
        "\n"
        "Endpoints:\n"
        "  GET  /health\n"
        "  GET  /v1/models\n"
        "  POST /v1/chat/completions   (supports \"stream\": true)\n",
        NITA_VERSION, argv0);
}

/* Strip a trailing .gguf and return a malloc'd id. */
static char *derive_model_id(const char *path)
{
    const char *base = nita_basename(path);
    size_t      len = strlen(base);
    char       *id;

    if (len > 5 && strcmp(base + len - 5, ".gguf") == 0) len -= 5;
    id = (char *)nita_malloc(len + 1);
    memcpy(id, base, len);
    id[len] = '\0';
    return id;
}

/* Exercise the compiled quantised kernel once so operators can see it work. */
static void kernel_probe(void)
{
    static float   scales[1]   = { 1.0f };
    static uint8_t a[2]        = { 0x55, 0xAA };
    static float   b_scales[1] = { 1.0f };
    static int32_t b[1]        = { 1 };
    int32_t        out = 0;

    BREADCRUMB("kernel", "probing compiled kernel");
    q4_matmul_kernel(scales, a, b_scales, b, &out, 1, 1, 1);
    printf("[nita] kernel (%s) probe: out=%d\n", NITA_QUANT_NAME, out);
}

/* Read all of stdin as UTF-8 bytes. */
static char *read_stdin_text(void)
{
    size_t cap = 256, len = 0;
    char  *buf = (char *)nita_malloc(cap);
    int    c;

    while ((c = getchar()) != EOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            buf = (char *)nita_realloc(buf, cap);
        }
        buf[len++] = (char)c;
    }
    buf[len] = '\0';
    return buf;
}

/* Tokenize a string, print the ids and verify the decode round-trip. */
static int run_tokenize(const tokenizer_t *tok, const char *arg)
{
    size_t   n = 0;
    int32_t *ids;
    char    *back;
    char    *owned = NULL;
    const char *text = arg;
    size_t   i;

    if (!tok || !tok->has_vocab) {
        fprintf(stderr, "nita: model has no vocabulary\n");
        return 1;
    }

    /* "-" reads the text from stdin, which is how non-ASCII input reaches
     * the engine on Windows, where argv is limited to the ANSI code page. */
    if (strcmp(arg, "-") == 0) {
        owned = read_stdin_text();
        text = owned;
    }

    ids = tokenizer_encode(tok, text, &n);

    printf("mode    : %s\n", tok->mode_name);
    printf("tokens  : %zu\n", n);
    printf("ids     :");
    for (i = 0; i < n; ++i) printf(" %d", ids[i]);
    printf("\n");

    back = tokenizer_decode(tok, ids, n);
    printf("decoded : \"%s\"\n", back ? back : "");
    printf("roundtrip: %s\n", (back && strcmp(back, text) == 0) ? "exact" : "DIFFERS");

    free(back);
    free(ids);
    free(owned);
    return 0;
}

/* Run the forward pass over explicit token ids and print the logits. */
static int run_logits_ids(const gguf_file_t *g, const int32_t *ids, size_t n)
{
    char          err[256];
    nita_model_t *m;
    float        *logits;
    int           nv;
    size_t        i;

    m = nita_model_open(g, 256, err, sizeof err);
    if (!m) {
        fprintf(stderr, "nita: %s\n", err);
        return 1;
    }

    nv = nita_model_n_vocab(m);
    logits = (float *)calloc((size_t)nv, sizeof(float));

    for (i = 0; i < n; ++i) {
        if (nita_forward(m, ids[i], logits) != 0) {
            fprintf(stderr, "nita: forward failed at token %zu\n", i);
            free(logits);
            nita_model_close(m);
            return 1;
        }
    }

    printf("tokens:");
    for (i = 0; i < n; ++i) printf(" %d", ids[i]);
    printf("\nlogits:");
    for (i = 0; i < (size_t)nv; ++i) printf(" %.6f", logits[i]);
    printf("\n");

    free(logits);
    nita_model_close(m);
    return 0;
}

/* Run the forward pass over a string and print the resulting logits.
 * Used to compare the C engine against the Python reference. */
static int run_logits(const gguf_file_t *g, const tokenizer_t *tok, const char *text)
{
    char          err[256];
    nita_model_t *m;
    int32_t      *ids = NULL;
    size_t        n = 0;
    float        *logits;
    int           nv;
    size_t        i;

    if (!tok || !tok->has_vocab) {
        fprintf(stderr, "nita: model has no vocabulary; cannot tokenize\n");
        return 1;
    }

    m = nita_model_open(g, 256, err, sizeof err);
    if (!m) {
        fprintf(stderr, "nita: %s\n", err);
        return 1;
    }

    ids = tokenizer_encode(tok, text, &n);
    if (!ids || n == 0) {
        fprintf(stderr, "nita: tokenization produced no tokens\n");
        nita_model_close(m);
        return 1;
    }

    nv = nita_model_n_vocab(m);
    logits = (float *)calloc((size_t)nv, sizeof(float));

    for (i = 0; i < n; ++i) {
        if (nita_forward(m, ids[i], logits) != 0) {
            fprintf(stderr, "nita: forward failed at token %zu\n", i);
            free(logits); free(ids); nita_model_close(m);
            return 1;
        }
    }

    printf("tokens:");
    for (i = 0; i < n; ++i) printf(" %d", ids[i]);
    printf("\nlogits:");
    for (i = 0; i < (size_t)nv; ++i) printf(" %.6f", logits[i]);
    printf("\n");

    free(logits);
    free(ids);
    nita_model_close(m);
    return 0;
}

static void inspect(const gguf_file_t *g, const tokenizer_t *tok)
{
    uint64_t i;

    printf("GGUF version    : %u\n", g->version);
    printf("file            : %s\n", g->path);
    printf("size            : %s\n", nita_human_bytes(g->file_size));
    printf("metadata keys   : %llu\n", (unsigned long long)g->kv_count);
    printf("tensors         : %llu\n", (unsigned long long)g->tensor_count);
    printf("data offset     : %llu\n", (unsigned long long)g->data_offset);
    printf("architecture    : %s\n", gguf_get_str(g, "general.architecture", "(none)"));
    printf("name            : %s\n", gguf_get_str(g, "general.name", "(none)"));

    if (tok && tok->has_vocab)
        printf("vocabulary      : %u pieces, mode=%s, merges=%u (bos=%d eos=%d unk=%d)\n",
               tok->n_tokens, tok->mode_name, tok->n_merges,
               tok->id_bos, tok->id_eos, tok->id_unk);
    else
        printf("vocabulary      : (absent)\n");

    printf("\nMetadata:\n");
    for (i = 0; i < g->kv_count; ++i) {
        const gguf_kv_t *kv = &g->kv[i];

        if (kv->type == GGUF_TYPE_STRING) {
            size_t n = (size_t)kv->v.str.len;
            if (n > 60)
                printf("  %-42s = \"<string, %zu bytes>\"\n", kv->key, n);
            else
                printf("  %-42s = \"%s\"\n", kv->key, kv->v.str.ptr);
        } else if (kv->type == GGUF_TYPE_ARRAY) {
            printf("  %-42s = array[%llu] of type %u\n", kv->key,
                   (unsigned long long)kv->count, kv->item_type);
        } else if (kv->type == GGUF_TYPE_FLOAT32 || kv->type == GGUF_TYPE_FLOAT64) {
            printf("  %-42s = %g\n", kv->key, kv->v.f64);
        } else if (kv->type == GGUF_TYPE_INT8 || kv->type == GGUF_TYPE_INT16 ||
                   kv->type == GGUF_TYPE_INT32 || kv->type == GGUF_TYPE_INT64) {
            printf("  %-42s = %lld\n", kv->key, (long long)kv->v.i64);
        } else {
            printf("  %-42s = %llu\n", kv->key, (unsigned long long)kv->v.u64);
        }
    }

    printf("\nTensors (first 40):\n");
    for (i = 0; i < g->tensor_count && i < 40; ++i) {
        const gguf_tensor_t *t = &g->tensors[i];
        uint32_t             d;

        printf("  %-40s %-8s dims=[", t->name, ggml_type_name(t->type));
        for (d = 0; d < t->n_dims; ++d)
            printf("%s%llu", d ? "," : "", (unsigned long long)t->dims[d]);
        printf("]\n");
    }
    if (g->tensor_count > 40)
        printf("  ... %llu more\n", (unsigned long long)(g->tensor_count - 40));
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *host = "0.0.0.0";
    const char *served_name = NULL;
    const char *quant_label = NULL;
    const char *logits_text = NULL;
    const char *tokens_arg = NULL;
    const char *tokenize_text = NULL;
    int         port = 7777;
    int         ctx = 2048;
    int         want_inspect = 0;
    int         i;

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            print_usage(stdout, argv[0]);
            return 0;
        } else if (strcmp(a, "--version") == 0) {
            printf("NITA %s (kernel: %s)\n", NITA_VERSION, NITA_QUANT_NAME);
            return 0;
        } else if (strcmp(a, "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(a, "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(a, "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(a, "--ctx") == 0 && i + 1 < argc) {
            ctx = atoi(argv[++i]);
        } else if (strcmp(a, "--name") == 0 && i + 1 < argc) {
            served_name = argv[++i];
        } else if (strcmp(a, "--quant-type") == 0 && i + 1 < argc) {
            quant_label = argv[++i];
        } else if (strcmp(a, "--inspect") == 0) {
            want_inspect = 1;
        } else if (strcmp(a, "--logits") == 0 && i + 1 < argc) {
            logits_text = argv[++i];
        } else if (strcmp(a, "--tokens") == 0 && i + 1 < argc) {
            tokens_arg = argv[++i];
        } else if (strcmp(a, "--tokenize") == 0 && i + 1 < argc) {
            tokenize_text = argv[++i];
        } else {
            fprintf(stderr, "nita: unknown argument '%s'\n\n", a);
            print_usage(stderr, argv[0]);
            return 2;
        }
    }

    if (!model_path) {
        fprintf(stderr, "nita: --model is required\n\n");
        print_usage(stderr, argv[0]);
        return 2;
    }
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "nita: --port must be between 1 and 65535\n");
        return 2;
    }
    if (ctx <= 0) ctx = 2048;

    {
        char          err[512];
        gguf_file_t  *g;
        tokenizer_t  *tok;
        nita_model_t *engine = NULL;
        char         *derived = NULL;
        nita_server_t srv;
        int           rc;

        printf("NITA %s (kernel: %s)\n", NITA_VERSION,
               quant_label ? quant_label : NITA_QUANT_NAME);
        printf("loading %s ...\n", model_path);
        fflush(stdout);

        g = gguf_load(model_path, err, sizeof err);
        if (!g) {
            fprintf(stderr, "nita: failed to load model: %s\n", err);
            return 1;
        }

        BREADCRUMB("model", "GGUF parsed");

        printf("loaded %s (%s, GGUF v%u, %llu tensors, %llu metadata keys)\n",
               nita_basename(g->path),
               nita_human_bytes(g->file_size),
               g->version,
               (unsigned long long)g->tensor_count,
               (unsigned long long)g->kv_count);

        tok = tokenizer_from_gguf(g);

        if (tokenize_text) {
            int trc = run_tokenize(tok, tokenize_text);
            tokenizer_free(tok);
            gguf_free(g);
            return trc;
        }

        if (want_inspect) {
            inspect(g, tok);
            tokenizer_free(tok);
            gguf_free(g);
            return 0;
        }

        if (tokens_arg) {
            int32_t ids[512];
            size_t  n = 0;
            char   *copy = nita_strdup(tokens_arg);
            char   *p = copy;
            int     lrc;

            while (*p && n < sizeof ids / sizeof ids[0]) {
                char *end;
                long  v = strtol(p, &end, 10);
                if (end == p) break;
                ids[n++] = (int32_t)v;
                p = end;
                while (*p == ',' || *p == ' ') ++p;
            }
            free(copy);

            if (n == 0) {
                fprintf(stderr, "nita: --tokens expects comma separated ids\n");
                tokenizer_free(tok);
                gguf_free(g);
                return 2;
            }
            lrc = run_logits_ids(g, ids, n);
            tokenizer_free(tok);
            gguf_free(g);
            return lrc;
        }

        if (logits_text) {
            int lrc = run_logits(g, tok, logits_text);
            tokenizer_free(tok);
            gguf_free(g);
            return lrc;
        }

        kernel_probe();
        SHERLOCK_HEADER();
        SHERLOCK_ROW("model-load", (int)g->tensor_count, 100.0, 0.0, 0.0, 0.0);

        /* Build the transformer. A metadata-only file has no usable weights,
         * in which case we say so and let the API fall back. */
        {
            char engine_err[256];
            engine = nita_model_open(g, ctx, engine_err, sizeof engine_err);
            if (engine)
                printf("transformer ready: ctx=%d vocab=%d\n",
                       nita_model_ctx(engine), nita_model_n_vocab(engine));
            else
                printf("transformer unavailable: %s\n",
                       engine_err[0] ? engine_err : "unknown reason");
            fflush(stdout);
        }

        memset(&srv, 0, sizeof srv);
        srv.model_path = model_path;
        srv.model      = g;
        srv.tok        = tok;
        srv.engine     = engine;
        srv.ctx_size   = ctx;
        srv.quant_name = quant_label ? quant_label : NITA_QUANT_NAME;

        if (served_name) {
            srv.model_id = served_name;
        } else {
            derived = derive_model_id(model_path);
            srv.model_id = derived;
        }

        printf("serving as model id '%s'\n", srv.model_id);
        fflush(stdout);

        rc = http_serve(host, port, &srv);

        nita_model_close(engine);
        free(derived);
        tokenizer_free(tok);
        gguf_free(g);
        return rc;
    }
}
