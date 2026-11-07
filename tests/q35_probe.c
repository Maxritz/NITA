/*
 * q35_probe.c – exercise the qwen35 gated delta-net block on a real
 * checkpoint's layer 0.
 *
 * This is a wiring and dimension check, not a correctness check: it proves
 * the tensor shapes derived from the metadata match what the file actually
 * stores, that the block executes, and that the recurrent state advances.
 * Numerical agreement with the reference still needs end-to-end logits.
 */

#include "nita.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const gguf_file_t *G;

static bool bind_name(nita_tensor_t *t, const char *name)
{
    uint64_t i;

    memset(t, 0, sizeof *t);
    for (i = 0; i < G->tensor_count; ++i) {
        const gguf_tensor_t *src = &G->tensors[i];
        size_t row, need;
        uint64_t abs_off;

        if (strcmp(src->name, name) != 0) continue;

        row = nita_row_bytes(src->type, src->n_dims > 0 ? src->dims[0] : 0);
        need = row * (size_t)(src->n_dims > 1 ? src->dims[1] : 1);
        abs_off = G->data_offset + src->offset;
        if (abs_off > G->file_size || need > G->file_size - abs_off) return false;

        t->data  = G->data + abs_off;
        t->type  = src->type;
        t->ne0   = src->n_dims > 0 ? src->dims[0] : 0;
        t->ne1   = src->n_dims > 1 ? src->dims[1] : 1;
        t->bound = true;
        return true;
    }
    return false;
}

static float normf(const float *v, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; ++i) s += (double)v[i] * v[i];
    return (float)sqrt(s);
}

static int finite_all(const float *v, int n)
{
    int i;
    for (i = 0; i < n; ++i) if (!isfinite(v[i])) return 0;
    return 1;
}

int main(int argc, char **argv)
{
    gguf_file_t *g;
    char   err[512];
    q35_gdn_cfg_t     cfg;
    q35_gdn_weights_t w;
    q35_gdn_state_t   st;
    float *x, *out;
    int    i, rc;

    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }

    g = gguf_load(argv[1], err, sizeof err);
    if (!g) { fprintf(stderr, "load failed: %s\n", err); return 1; }
    G = g;

    memset(&cfg, 0, sizeof cfg);
    cfg.n_embd      = (int)gguf_get_int(g, "qwen35.embedding_length", 0);
    cfg.n_k_heads   = (int)gguf_get_int(g, "qwen35.ssm.group_count", 0);
    cfg.head_k_dim  = (int)gguf_get_int(g, "qwen35.ssm.state_size", 0);
    cfg.n_v_heads   = (int)gguf_get_int(g, "qwen35.ssm.time_step_rank", 0);
    cfg.conv_kernel = (int)gguf_get_int(g, "qwen35.ssm.conv_kernel", 0);
    cfg.eps         = (float)gguf_get_float(g, "qwen35.attention.layer_norm_rms_epsilon", 1e-6);
    {
        int inner = (int)gguf_get_int(g, "qwen35.ssm.inner_size", 0);
        cfg.head_v_dim = cfg.n_v_heads > 0 ? inner / cfg.n_v_heads : 0;
        cfg.conv_dim   = 2 * cfg.n_k_heads * cfg.head_k_dim + inner;
    }

    printf("cfg: n_embd=%d n_k_heads=%d head_k_dim=%d n_v_heads=%d "
           "head_v_dim=%d conv_kernel=%d conv_dim=%d eps=%g\n",
           cfg.n_embd, cfg.n_k_heads, cfg.head_k_dim, cfg.n_v_heads,
           cfg.head_v_dim, cfg.conv_kernel, cfg.conv_dim, (double)cfg.eps);

    if (!bind_name(&w.in_norm,   "blk.0.attn_norm.weight") ||
        !bind_name(&w.qkv,      "blk.0.attn_qkv.weight") ||
        !bind_name(&w.gate,     "blk.0.attn_gate.weight") ||
        !bind_name(&w.conv1d,   "blk.0.ssm_conv1d.weight") ||
        !bind_name(&w.alpha,    "blk.0.ssm_alpha.weight") ||
        !bind_name(&w.beta,     "blk.0.ssm_beta.weight") ||
        !bind_name(&w.dt_bias,  "blk.0.ssm_dt.bias") ||
        !bind_name(&w.ssm_a,    "blk.0.ssm_a") ||
        !bind_name(&w.ssm_norm, "blk.0.ssm_norm.weight") ||
        !bind_name(&w.out,      "blk.0.ssm_out.weight") ||
        !bind_name(&w.post_norm,"blk.0.post_attention_norm.weight")) {
        fprintf(stderr, "FAIL: a blk.0 gated delta-net tensor is missing\n");
        return 1;
    }

    printf("tensors: qkv=[%d,%d] gate=[%d,%d] conv1d=[%d,%d] out=[%d,%d] ssm_norm=[%d]\n",
           (int)w.qkv.ne0, (int)w.qkv.ne1, (int)w.gate.ne0, (int)w.gate.ne1,
           (int)w.conv1d.ne0, (int)w.conv1d.ne1, (int)w.out.ne0, (int)w.out.ne1,
           (int)w.ssm_norm.ne0);

    rc = q35_gdn_prepare(&cfg, &w, &st, err, sizeof err);
    if (rc != 0) { fprintf(stderr, "prepare failed: %s\n", err); return 1; }

    x   = (float *)malloc((size_t)cfg.n_embd * sizeof(float));
    out = (float *)malloc((size_t)cfg.n_embd * sizeof(float));

    /* Deterministic pseudo-random residual, scaled like a real hidden state. */
    for (i = 0; i < cfg.n_embd; ++i) {
        unsigned s = (unsigned)(i * 2654435761u + 12345u);
        s ^= s >> 13; s *= 1274126177u; s ^= s >> 16;
        x[i] = ((float)(s & 0xFFFF) / 32768.0f - 1.0f) * 0.5f;
    }

    for (int step = 0; step < 3; ++step) {
        rc = q35_gdn_layer(&cfg, &w, &st, x, out);
        if (rc != 0) { fprintf(stderr, "layer step %d failed rc=%d\n", step, rc); return 1; }
        printf("step %d: in_norm=%.4f out_norm=%.4f finite=%d out[0..3]=%.5f %.5f %.5f %.5f\n",
               step, normf(x, cfg.n_embd), normf(out, cfg.n_embd),
               finite_all(out, cfg.n_embd), out[0], out[1], out[2], out[3]);
        printf("        conv_state_norm=%.4f delta_state_norm=%.4f\n",
               normf(st.conv_state, (cfg.conv_kernel - 1) * cfg.conv_dim),
               normf(st.delta, cfg.n_v_heads * cfg.head_k_dim * cfg.head_v_dim));
    }

    q35_gdn_state_free(&st);
    free(x); free(out);
    gguf_free(g);
    return 0;
}
