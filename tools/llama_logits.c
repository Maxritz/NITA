/*
 * Dump llama.cpp's logits for an explicit token sequence.
 *
 * NITA's --tokens mode prints the same thing, so this is what makes a
 * like-for-like comparison possible: both sides are given token ids (no
 * tokeniser in the loop) and both print the full vocabulary distribution for
 * the last position. llama-cli in the released binaries has no flag that dumps
 * raw logits, only sampled text, hence this little driver against llama.dll.
 *
 * Build from the directory holding the llama.cpp release DLLs and headers:
 *
 *   gcc -O2 -I. ../../tools/llama_logits.c -o llama_logits.exe \
 *       -L. -l:llama.dll -l:ggml.dll -l:ggml-base.dll
 *
 * Usage:
 *
 *   ./llama_logits.exe model.gguf 3,4,5 [f32kv]
 *
 * llama.cpp defaults to an f16 KV cache; passing f32kv selects f32 for both K
 * and V so a comparison against an engine that keeps them in f32 is not
 * measuring KV precision rather than the forward pass.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

#define MAX_TOKENS 1024

/* Opt-in per-node dump (NITA_FORK_DUMP=path). The fork tags every block's
 * residual stream "l_out-<il>" and the final norm "result_norm", which is
 * exactly the sequence NITA's own NITA_Q35_DUMP prints, so the two files can be
 * compared layer by layer. */
static FILE *g_dump;   /* NULL unless NITA_FORK_DUMP is set */

/* Walks the whole logical tensor, not just ne[0]: the block taps are views
 * ([head_dim, n_heads, n_tokens] and friends), so reading ne[0] straight from
 * t->data would hand back the first head only. */
static void dump_tensor(const struct ggml_tensor *t)
{
    const char *base = (const char *)t->data;
    int64_t     ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    int64_t     i0, i1, i2, i3;

    if (!base || t->type != GGML_TYPE_F32) return;

    fprintf(g_dump, "%s %lld", t->name, (long long)(ne0 * ne1 * ne2 * ne3));
    for (i3 = 0; i3 < ne3; ++i3)
    for (i2 = 0; i2 < ne2; ++i2)
    for (i1 = 0; i1 < ne1; ++i1)
    for (i0 = 0; i0 < ne0; ++i0) {
        const float *p = (const float *)(base + i0 * t->nb[0] + i1 * t->nb[1] +
                                         i2 * t->nb[2] + i3 * t->nb[3]);
        fprintf(g_dump, " %.6g", *p);
    }
    fputc('\n', g_dump);
    fflush(g_dump);
}

static bool dump_cb(struct ggml_tensor *t, bool ask, void *ud)
{
    static const char *prefixes[] = {
        "l_out-", "attn_norm-", "linear_attn_out-", "ffn_out-", "attn_residual-",
        /* mixer internals the fork already tags */
        "conv_output_silu-", "q_conv-", "k_conv-", "v_conv-", "beta_sigmoid-",
        "a_softplus-", "z-", "attn_out_norm-", "final_output-", "qkv_mixed-",
        "q_conv_predelta-", "v_conv_predelta-", "gate-"
    };
    size_t i;

    (void)ud;
    if (ask) return true;
    for (i = 0; i < sizeof prefixes / sizeof prefixes[0]; ++i)
        if (strncmp(t->name, prefixes[i], strlen(prefixes[i])) == 0) { dump_tensor(t); return true; }
    if (strcmp(t->name, "result_norm") == 0) dump_tensor(t);
    return true;
}

int main(int argc, char **argv)
{
    llama_token ids[MAX_TOKENS];
    int         n = 0;
    char       *tok;

    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf id,id,...\n", argv[0]);
        return 2;
    }

    for (tok = strtok(argv[2], ","); tok != NULL && n < MAX_TOKENS; tok = strtok(NULL, ","))
        ids[n++] = (llama_token)atoi(tok);
    if (n == 0) {
        fprintf(stderr, "no token ids given\n");
        return 2;
    }

    llama_backend_init();

    {
        struct llama_model_params mp = llama_model_default_params();
        struct llama_model       *model;
        struct llama_context     *ctx;
        const struct llama_vocab *vocab;
        const float              *logits;
        int                       n_vocab;
        int                       i;

        mp.n_gpu_layers = 0;

        model = llama_model_load_from_file(argv[1], mp);
        if (!model) {
            fprintf(stderr, "failed to load %s\n", argv[1]);
            llama_backend_free();
            return 1;
        }

        {
            struct llama_context_params cp = llama_context_default_params();

            cp.n_ctx     = (uint32_t)(n + 8);
            cp.n_batch   = (uint32_t)n;
            cp.n_ubatch  = (uint32_t)n;
            cp.no_perf   = true;
            {
                const char *dump = getenv("NITA_FORK_DUMP");

                if (dump && *dump) {
                    g_dump = fopen(dump, "w");
                    if (!g_dump) { fprintf(stderr, "cannot open %s\n", dump); return 2; }
                    cp.cb_eval           = dump_cb;
                    cp.cb_eval_user_data = NULL;
                }
            }
            if (argc > 3 && strcmp(argv[3], "f32kv") == 0) {
                cp.type_k = GGML_TYPE_F32;
                cp.type_v = GGML_TYPE_F32;
            }
            ctx = llama_init_from_model(model, cp);
        }
        if (!ctx) {
            fprintf(stderr, "failed to create context\n");
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        {
            struct llama_batch batch = llama_batch_get_one(ids, n);

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "llama_decode failed\n");
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }

        logits  = llama_get_logits_ith(ctx, -1);
        vocab   = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);

        printf("tokens:");
        for (i = 0; i < n; ++i) printf(" %d", (int)ids[i]);
        printf("\nlogits:");
        for (i = 0; i < n_vocab; ++i) printf(" %.6f", (double)logits[i]);
        printf("\n");

        llama_free(ctx);
        llama_model_free(model);
    }

    llama_backend_free();
    return 0;
}
