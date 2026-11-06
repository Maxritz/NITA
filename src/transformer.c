/*
 * transformer.c – llama-style decoder forward pass.
 *
 * Architecture implemented (GGUF tensor naming):
 *
 *   token_embd.weight                    [n_embd, n_vocab]
 *   output_norm.weight                   [n_embd]
 *   output.weight                        [n_embd, n_vocab]   (optional; tied to token_embd)
 *   blk.N.attn_norm.weight               [n_embd]
 *   blk.N.attn_q.weight                  [n_embd, n_head*head_dim]
 *   blk.N.attn_k.weight                  [n_embd, n_head_kv*head_dim]
 *   blk.N.attn_v.weight                  [n_embd, n_head_kv*head_dim]
 *   blk.N.attn_output.weight             [n_head*head_dim, n_embd]
 *   blk.N.ffn_norm.weight                [n_embd]
 *   blk.N.ffn_gate.weight                [n_embd, n_ff]
 *   blk.N.ffn_up.weight                  [n_embd, n_ff]
 *   blk.N.ffn_down.weight                [n_ff, n_embd]
 *
 * Per token: embedding lookup, RMSNorm, single/multi-head attention with a
 * KV cache and grouped-query heads, RoPE on Q and K, SwiGLU feed-forward,
 * final RMSNorm and the output projection.
 *
 * Weights stay in their GGUF quantised form; each row is dequantised into a
 * reusable buffer immediately before use, so no full f32 matrix is created.
 */

#include "nita.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct nita_model {
    /* Hyperparameters */
    int   n_embd;
    int   n_layer;
    int   n_head;
    int   n_head_kv;
    int   n_ff;
    int   n_rot;
    int   n_vocab;
    int   ctx;
    int   head_dim;
    int   kv_dim;
    float rms_eps;
    float rope_theta;
    bool  tied_output;
    bool  ready;

    /* Weights */
    nita_tensor_t  tok_embd;
    nita_tensor_t  out_norm;
    nita_tensor_t  out_proj;
    nita_tensor_t *attn_norm;
    nita_tensor_t *attn_q;
    nita_tensor_t *attn_k;
    nita_tensor_t *attn_v;
    nita_tensor_t *attn_out;
    nita_tensor_t *ffn_norm;
    nita_tensor_t *ffn_gate;
    nita_tensor_t *ffn_up;
    nita_tensor_t *ffn_down;

    /* KV cache: one contiguous buffer per layer */
    float **k_cache;
    float **v_cache;
    int    pos;

    /* Scratch */
    float  *rowbuf;
    size_t  rowcap;
    float  *nbuf;    /* dequantised norm vector      */
    float  *xb;      /* residual stream              */
    float  *hb;      /* normalised hidden            */
    float  *qbuf;    /* queries                      */
    float  *kbuf;    /* keys for this token          */
    float  *vbuf;    /* values for this token        */
    float  *abuf;    /* attention output             */
    float  *gate;
    float  *up;
    float  *ff;
    float  *tmp;
    float  *scores;
};

/* ------------------------------------------------------------------ */
/* Tensor binding                                                      */
/* ------------------------------------------------------------------ */
static bool bound_in_file(const gguf_file_t *g, const nita_tensor_t *t)
{
    (void)g;
    return t->bound && t->data != NULL;
}

static bool bind_tensor(const gguf_file_t *g, nita_tensor_t *t, const char *name)
{
    uint64_t i;

    memset(t, 0, sizeof *t);
    if (!g || !name) return false;

    for (i = 0; i < g->tensor_count; ++i) {
        const gguf_tensor_t *src = &g->tensors[i];
        size_t               row, need;
        uint64_t             abs_off;

        if (strcmp(src->name, name) != 0) continue;

        row  = nita_row_bytes(src->type, src->n_dims > 0 ? src->dims[0] : 0);
        need = row * (size_t)(src->n_dims > 1 ? src->dims[1] : 1);
        abs_off = g->data_offset + src->offset;

        if (abs_off > g->file_size || need > g->file_size - abs_off) {
            BREADCRUMBF("model", "tensor '%s' runs past end of file", name);
            return false;
        }

        t->data  = g->data + abs_off;
        t->type  = src->type;
        t->ne0   = src->n_dims > 0 ? src->dims[0] : 0;
        t->ne1   = src->n_dims > 1 ? src->dims[1] : 1;
        t->bound = true;
        return true;
    }
    return false;
}

/* Read a hyperparameter, trying the model's own architecture prefix first. */
static int64_t hparam_int(const gguf_file_t *g, const char *arch, const char *suffix,
                          int64_t fallback)
{
    char key[160];
    if (arch && *arch) {
        snprintf(key, sizeof key, "%s.%s", arch, suffix);
        if (gguf_has_key(g, key)) return gguf_get_int(g, key, fallback);
    }
    snprintf(key, sizeof key, "llama.%s", suffix);
    if (gguf_has_key(g, key)) return gguf_get_int(g, key, fallback);
    return fallback;
}

static double hparam_float(const gguf_file_t *g, const char *arch, const char *suffix,
                           double fallback)
{
    char key[160];
    if (arch && *arch) {
        snprintf(key, sizeof key, "%s.%s", arch, suffix);
        if (gguf_has_key(g, key)) return gguf_get_float(g, key, fallback);
    }
    snprintf(key, sizeof key, "llama.%s", suffix);
    if (gguf_has_key(g, key)) return gguf_get_float(g, key, fallback);
    return fallback;
}

/* ------------------------------------------------------------------ */
/* Small numeric kernels                                               */
/* ------------------------------------------------------------------ */
/* Dequantise a 1-D tensor (a norm weight vector) into dst. */
static int load_vector(const nita_tensor_t *t, float *dst, int n)
{
    if (!bound_in_file(NULL, t)) return -1;
    return nita_dequant_row(t->data, t->type, (uint64_t)n, dst);
}

/* y = W x, where W is stored row-major as ne1 rows of ne0 elements. */
static int matvec(nita_model_t *m, const nita_tensor_t *t,
                  const float *x, float *y, int expect_in, int expect_out)
{
    uint64_t r;
    size_t   row_bytes;

    if (!bound_in_file(NULL, t)) return -1;
    if ((int)t->ne0 != expect_in || (int)t->ne1 != expect_out) return -2;

    row_bytes = nita_row_bytes(t->type, t->ne0);

    for (r = 0; r < t->ne1; ++r) {
        const uint8_t *src = t->data + (size_t)r * row_bytes;
        const float   *w;
        float          sum = 0.0f;
        uint64_t       i;

        if (nita_dequant_row(src, t->type, t->ne0, m->rowbuf) != 0) return -3;
        w = m->rowbuf;
        for (i = 0; i < t->ne0; ++i) sum += w[i] * x[i];
        y[r] = sum;
    }
    return 0;
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps)
{
    double ss = 0.0;
    int    i;
    float  scale;

    for (i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    ss /= (double)n;
    scale = 1.0f / sqrtf((float)ss + eps);
    for (i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

static float silu(float x)
{
    return x / (1.0f + expf(-x));
}

static void softmax_inplace(float *v, int n)
{
    float maxv = v[0];
    float sum = 0.0f;
    int   i;

    for (i = 1; i < n; ++i) if (v[i] > maxv) maxv = v[i];
    for (i = 0; i < n; ++i) { v[i] = expf(v[i] - maxv); sum += v[i]; }
    if (sum <= 0.0f) sum = 1.0f;
    for (i = 0; i < n; ++i) v[i] /= sum;
}

/*
 * Rotary position embedding, adjacent-pair layout.
 *
 * GGUF llama checkpoints are stored with Q/K rows permuted by the converter
 * precisely so that this interleaved form (rotating x[2i] against x[2i+1])
 * matches the original model.
 */
static void rope_apply(float *vec, int n_heads, int head_dim, int n_rot,
                       int pos, float theta_base)
{
    int h, i;

    for (h = 0; h < n_heads; ++h) {
        float *p = vec + (size_t)h * (size_t)head_dim;
        for (i = 0; i < n_rot / 2; ++i) {
            float freq  = powf(theta_base, -2.0f * (float)i / (float)n_rot);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = p[2 * i];
            float x1 = p[2 * i + 1];
            p[2 * i]     = x0 * c - x1 * s;
            p[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */
static void free_model(nita_model_t *m)
{
    int l;
    if (!m) return;

    if (m->k_cache) {
        for (l = 0; l < m->n_layer; ++l) free(m->k_cache[l]);
        free(m->k_cache);
    }
    if (m->v_cache) {
        for (l = 0; l < m->n_layer; ++l) free(m->v_cache[l]);
        free(m->v_cache);
    }

    free(m->attn_norm); free(m->attn_q); free(m->attn_k); free(m->attn_v);
    free(m->attn_out);  free(m->ffn_norm); free(m->ffn_gate);
    free(m->ffn_up);    free(m->ffn_down);

    free(m->rowbuf); free(m->nbuf); free(m->xb); free(m->hb);
    free(m->qbuf); free(m->kbuf); free(m->vbuf); free(m->abuf);
    free(m->gate); free(m->up); free(m->ff); free(m->tmp); free(m->scores);

    free(m);
}

void nita_model_close(nita_model_t *m)
{
    free_model(m);
}

nita_model_t *nita_model_open(const gguf_file_t *g, int ctx, char *err, size_t errlen)
{
    nita_model_t *m;
    const char   *arch;
    int           l, max_row;
    int           q_dim;

    if (err && errlen) err[0] = '\0';

    if (!g) {
        snprintf(err, errlen, "no model file");
        return NULL;
    }

    m = (nita_model_t *)nita_calloc(1, sizeof *m);
    arch = gguf_get_str(g, "general.architecture", "llama");

    m->n_layer   = (int)hparam_int(g, arch, "block_count", 0);
    m->n_embd    = (int)hparam_int(g, arch, "embedding_length", 0);
    m->n_head    = (int)hparam_int(g, arch, "attention.head_count", 0);
    m->n_head_kv = (int)hparam_int(g, arch, "attention.head_count_kv", m->n_head);
    m->n_ff      = (int)hparam_int(g, arch, "feed_forward_length", 0);
    m->rms_eps   = (float)hparam_float(g, arch, "attention.layer_norm_rms_epsilon", 1e-5);
    m->rope_theta= (float)hparam_float(g, arch, "rope.freq_base", 10000.0);

    if (m->n_layer <= 0 || m->n_embd <= 0 || m->n_head <= 0) {
        snprintf(err, errlen,
                 "not a transformer: block_count=%d embedding_length=%d head_count=%d",
                 m->n_layer, m->n_embd, m->n_head);
        free_model(m);
        return NULL;
    }
    if (m->n_head_kv <= 0) m->n_head_kv = m->n_head;
    if (m->n_head_kv > m->n_head) m->n_head_kv = m->n_head;
    if (m->n_embd % m->n_head != 0) {
        snprintf(err, errlen, "embedding_length %d is not divisible by head_count %d",
                 m->n_embd, m->n_head);
        free_model(m);
        return NULL;
    }

    m->head_dim = m->n_embd / m->n_head;
    m->kv_dim   = m->head_dim * m->n_head_kv;
    m->n_rot    = (int)hparam_int(g, arch, "rope.dimension_count", m->head_dim);
    if (m->n_rot <= 0 || m->n_rot > m->head_dim) m->n_rot = m->head_dim;
    if (m->n_rot % 2 != 0) m->n_rot -= 1;

    if (m->n_ff <= 0) {
        snprintf(err, errlen, "feed_forward_length missing");
        free_model(m);
        return NULL;
    }

    m->ctx = ctx > 0 ? ctx : 2048;

    /* Vocabulary size comes from the embedding matrix; fall back to metadata. */
    if (!bind_tensor(g, &m->tok_embd, "token_embd.weight")) {
        snprintf(err, errlen, "missing tensor 'token_embd.weight'");
        free_model(m);
        return NULL;
    }
    m->n_vocab = (int)m->tok_embd.ne1;
    if (m->n_vocab <= 0)
        m->n_vocab = (int)hparam_int(g, arch, "vocab_size", 0);
    if (m->n_vocab <= 0) {
        snprintf(err, errlen, "cannot determine vocabulary size");
        free_model(m);
        return NULL;
    }

    if (!bind_tensor(g, &m->out_norm, "output_norm.weight")) {
        snprintf(err, errlen, "missing tensor 'output_norm.weight'");
        free_model(m);
        return NULL;
    }

    m->tied_output = !bind_tensor(g, &m->out_proj, "output.weight");
    if (m->tied_output) m->out_proj = m->tok_embd;

    /* Per-layer tensors */
    m->attn_norm = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_norm);
    m->attn_q    = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_q);
    m->attn_k    = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_k);
    m->attn_v    = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_v);
    m->attn_out  = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->attn_out);
    m->ffn_norm  = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_norm);
    m->ffn_gate  = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_gate);
    m->ffn_up    = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_up);
    m->ffn_down  = (nita_tensor_t *)nita_calloc((size_t)m->n_layer, sizeof *m->ffn_down);

    for (l = 0; l < m->n_layer; ++l) {
        char name[96];

        snprintf(name, sizeof name, "blk.%d.attn_norm.weight", l);
        if (!bind_tensor(g, &m->attn_norm[l], name)) goto missing_layer;
        snprintf(name, sizeof name, "blk.%d.attn_q.weight", l);
        if (!bind_tensor(g, &m->attn_q[l], name)) goto missing_layer;
        snprintf(name, sizeof name, "blk.%d.attn_k.weight", l);
        if (!bind_tensor(g, &m->attn_k[l], name)) goto missing_layer;
        snprintf(name, sizeof name, "blk.%d.attn_v.weight", l);
        if (!bind_tensor(g, &m->attn_v[l], name)) goto missing_layer;
        snprintf(name, sizeof name, "blk.%d.attn_output.weight", l);
        if (!bind_tensor(g, &m->attn_out[l], name)) goto missing_layer;
        snprintf(name, sizeof name, "blk.%d.ffn_norm.weight", l);
        if (!bind_tensor(g, &m->ffn_norm[l], name)) goto missing_layer;
        snprintf(name, sizeof name, "blk.%d.ffn_gate.weight", l);
        if (!bind_tensor(g, &m->ffn_gate[l], name)) goto missing_layer;
        snprintf(name, sizeof name, "blk.%d.ffn_up.weight", l);
        if (!bind_tensor(g, &m->ffn_up[l], name)) goto missing_layer;
        snprintf(name, sizeof name, "blk.%d.ffn_down.weight", l);
        if (!bind_tensor(g, &m->ffn_down[l], name)) goto missing_layer;
    }

    /* Shape consistency */
    q_dim = m->n_head * m->head_dim;
    for (l = 0; l < m->n_layer; ++l) {
        if ((int)m->attn_q[l].ne0 != m->n_embd || (int)m->attn_q[l].ne1 != q_dim ||
            (int)m->attn_k[l].ne0 != m->n_embd || (int)m->attn_k[l].ne1 != m->kv_dim ||
            (int)m->attn_v[l].ne0 != m->n_embd || (int)m->attn_v[l].ne1 != m->kv_dim ||
            (int)m->attn_out[l].ne0 != q_dim || (int)m->attn_out[l].ne1 != m->n_embd ||
            (int)m->ffn_gate[l].ne0 != m->n_embd || (int)m->ffn_gate[l].ne1 != m->n_ff ||
            (int)m->ffn_up[l].ne0 != m->n_embd || (int)m->ffn_up[l].ne1 != m->n_ff ||
            (int)m->ffn_down[l].ne0 != m->n_ff || (int)m->ffn_down[l].ne1 != m->n_embd) {
            snprintf(err, errlen, "layer %d tensor shapes are inconsistent", l);
            free_model(m);
            return NULL;
        }
    }

    /* Scratch buffers */
    max_row = m->n_embd;
    if (m->n_ff > max_row) max_row = m->n_ff;
    m->rowbuf = (float *)nita_calloc((size_t)max_row, sizeof(float));
    m->rowcap = (size_t)max_row;

    m->nbuf  = (float *)nita_calloc((size_t)m->n_embd, sizeof(float));
    m->xb    = (float *)nita_calloc((size_t)m->n_embd, sizeof(float));
    m->hb    = (float *)nita_calloc((size_t)m->n_embd, sizeof(float));
    m->qbuf  = (float *)nita_calloc((size_t)q_dim, sizeof(float));
    m->kbuf  = (float *)nita_calloc((size_t)m->kv_dim, sizeof(float));
    m->vbuf  = (float *)nita_calloc((size_t)m->kv_dim, sizeof(float));
    m->abuf  = (float *)nita_calloc((size_t)q_dim, sizeof(float));
    m->gate  = (float *)nita_calloc((size_t)m->n_ff, sizeof(float));
    m->up    = (float *)nita_calloc((size_t)m->n_ff, sizeof(float));
    m->ff    = (float *)nita_calloc((size_t)m->n_ff, sizeof(float));
    m->tmp   = (float *)nita_calloc((size_t)m->n_embd, sizeof(float));
    m->scores= (float *)nita_calloc((size_t)m->ctx, sizeof(float));

    /* KV cache */
    m->k_cache = (float **)nita_calloc((size_t)m->n_layer, sizeof(float *));
    m->v_cache = (float **)nita_calloc((size_t)m->n_layer, sizeof(float *));
    for (l = 0; l < m->n_layer; ++l) {
        size_t elems = (size_t)m->ctx * (size_t)m->kv_dim;
        m->k_cache[l] = (float *)nita_calloc(elems, sizeof(float));
        m->v_cache[l] = (float *)nita_calloc(elems, sizeof(float));
    }

    m->pos   = 0;
    m->ready = true;

    BREADCRUMBF("model",
                "transformer: n_embd=%d n_layer=%d n_head=%d n_head_kv=%d "
                "n_ff=%d n_rot=%d n_vocab=%d ctx=%d tied_output=%d",
                m->n_embd, m->n_layer, m->n_head, m->n_head_kv,
                m->n_ff, m->n_rot, m->n_vocab, m->ctx, (int)m->tied_output);
    return m;

missing_layer:
    snprintf(err, errlen, "layer %d is missing required tensors", l);
    free_model(m);
    return NULL;
}

void nita_model_reset(nita_model_t *m)
{
    int l;
    if (!m) return;
    for (l = 0; l < m->n_layer; ++l) {
        size_t bytes = (size_t)m->ctx * (size_t)m->kv_dim * sizeof(float);
        memset(m->k_cache[l], 0, bytes);
        memset(m->v_cache[l], 0, bytes);
    }
    m->pos = 0;
}

int nita_model_ctx(const nita_model_t *m)      { return m ? m->ctx : 0; }
int nita_model_n_vocab(const nita_model_t *m)  { return m ? m->n_vocab : 0; }

/* ------------------------------------------------------------------ */
/* Forward pass                                                        */
/* ------------------------------------------------------------------ */
static int attention(nita_model_t *m, int layer, int pos)
{
    int   h;
    float inv = 1.0f / sqrtf((float)m->head_dim);
    float *kcache = m->k_cache[layer];
    float *vcache = m->v_cache[layer];

    for (h = 0; h < m->n_head; ++h) {
        int         kvh = (int)((long)h * m->n_head_kv / m->n_head);
        const float *qh = m->qbuf + (size_t)h * (size_t)m->head_dim;
        float       *oh = m->abuf + (size_t)h * (size_t)m->head_dim;
        int         t, d;

        for (t = 0; t <= pos; ++t) {
            const float *kt = kcache + (size_t)t * (size_t)m->kv_dim
                            + (size_t)kvh * (size_t)m->head_dim;
            float sum = 0.0f;
            for (d = 0; d < m->head_dim; ++d) sum += qh[d] * kt[d];
            m->scores[t] = sum * inv;
        }
        softmax_inplace(m->scores, pos + 1);

        for (d = 0; d < m->head_dim; ++d) oh[d] = 0.0f;
        for (t = 0; t <= pos; ++t) {
            const float *vt = vcache + (size_t)t * (size_t)m->kv_dim
                            + (size_t)kvh * (size_t)m->head_dim;
            float p = m->scores[t];
            for (d = 0; d < m->head_dim; ++d) oh[d] += p * vt[d];
        }
    }
    return 0;
}

int nita_forward(nita_model_t *m, int32_t token, float *logits)
{
    int   l;
    int   pos;
    int   q_dim;
    size_t row_bytes;

    if (!m || !m->ready) return -1;
    if (token < 0 || token >= m->n_vocab) return -2;
    if (m->pos >= m->ctx) return -3;

    pos   = m->pos;
    q_dim = m->n_head * m->head_dim;

    /* Token embedding lookup */
    row_bytes = nita_row_bytes(m->tok_embd.type, (uint64_t)m->n_embd);
    if (nita_dequant_row(m->tok_embd.data + (size_t)token * row_bytes,
                         m->tok_embd.type, (uint64_t)m->n_embd, m->xb) != 0)
        return -4;

    for (l = 0; l < m->n_layer; ++l) {
        /* ---- attention block ---- */
        if (load_vector(&m->attn_norm[l], m->nbuf, m->n_embd) != 0) return -5;
        rmsnorm(m->hb, m->xb, m->nbuf, m->n_embd, m->rms_eps);

        if (matvec(m, &m->attn_q[l], m->hb, m->qbuf, m->n_embd, q_dim) != 0) return -6;
        if (matvec(m, &m->attn_k[l], m->hb, m->kbuf, m->n_embd, m->kv_dim) != 0) return -6;
        if (matvec(m, &m->attn_v[l], m->hb, m->vbuf, m->n_embd, m->kv_dim) != 0) return -6;

        rope_apply(m->qbuf, m->n_head,    m->head_dim, m->n_rot, pos, m->rope_theta);
        rope_apply(m->kbuf, m->n_head_kv, m->head_dim, m->n_rot, pos, m->rope_theta);

        memcpy(m->k_cache[l] + (size_t)pos * (size_t)m->kv_dim,
               m->kbuf, (size_t)m->kv_dim * sizeof(float));
        memcpy(m->v_cache[l] + (size_t)pos * (size_t)m->kv_dim,
               m->vbuf, (size_t)m->kv_dim * sizeof(float));

        if (attention(m, l, pos) != 0) return -7;
        if (matvec(m, &m->attn_out[l], m->abuf, m->tmp, q_dim, m->n_embd) != 0) return -8;

        {
            int i;
            for (i = 0; i < m->n_embd; ++i) m->xb[i] += m->tmp[i];
        }

        /* ---- feed-forward block (SwiGLU) ---- */
        if (load_vector(&m->ffn_norm[l], m->nbuf, m->n_embd) != 0) return -5;
        rmsnorm(m->hb, m->xb, m->nbuf, m->n_embd, m->rms_eps);

        if (matvec(m, &m->ffn_gate[l], m->hb, m->gate, m->n_embd, m->n_ff) != 0) return -9;
        if (matvec(m, &m->ffn_up[l],   m->hb, m->up,   m->n_embd, m->n_ff) != 0) return -9;

        {
            int i;
            for (i = 0; i < m->n_ff; ++i) m->ff[i] = silu(m->gate[i]) * m->up[i];
        }

        if (matvec(m, &m->ffn_down[l], m->ff, m->tmp, m->n_ff, m->n_embd) != 0) return -10;

        {
            int i;
            for (i = 0; i < m->n_embd; ++i) m->xb[i] += m->tmp[i];
        }
    }

    /* Final norm and output projection */
    if (load_vector(&m->out_norm, m->nbuf, m->n_embd) != 0) return -5;
    rmsnorm(m->hb, m->xb, m->nbuf, m->n_embd, m->rms_eps);

    if (matvec(m, &m->out_proj, m->hb, logits, m->n_embd, m->n_vocab) != 0) return -11;

    m->pos = pos + 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Sampling                                                            */
/* ------------------------------------------------------------------ */
static uint32_t rng_next(uint32_t *s)
{
    uint32_t x = *s ? *s : 0x9E3779B9u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

typedef struct { float v; int32_t i; } cand_t;

static int cand_cmp(const void *a, const void *b)
{
    const cand_t *x = (const cand_t *)a;
    const cand_t *y = (const cand_t *)b;
    if (x->v < y->v) return 1;
    if (x->v > y->v) return -1;
    return 0;
}

int32_t nita_sample(const float *logits, int n_vocab,
                    float temperature, int top_k, uint32_t *rng)
{
    cand_t  *cand;
    int      k, i;
    float    maxlogit;
    double   total = 0.0;
    double   target;
    int32_t  chosen;

    if (!logits || n_vocab <= 0) return 0;

    /* Greedy when temperature is zero or negative. */
    if (temperature <= 0.0f) {
        int32_t best = 0;
        for (i = 1; i < n_vocab; ++i)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    cand = (cand_t *)nita_malloc((size_t)n_vocab * sizeof *cand);
    maxlogit = logits[0];
    for (i = 0; i < n_vocab; ++i) {
        cand[i].v = logits[i];
        cand[i].i = i;
        if (logits[i] > maxlogit) maxlogit = logits[i];
    }

    k = (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab;
    if (k < n_vocab) {
        /* Partial selection is enough: sort then keep the head. */
        qsort(cand, (size_t)n_vocab, sizeof *cand, cand_cmp);
    } else {
        qsort(cand, (size_t)n_vocab, sizeof *cand, cand_cmp);
    }

    /* Convert to probabilities over the retained set. */
    {
        double *p = (double *)nita_malloc((size_t)k * sizeof *p);
        for (i = 0; i < k; ++i) {
            p[i] = exp((double)((cand[i].v - maxlogit) / temperature));
            total += p[i];
        }
        if (total <= 0.0) total = 1.0;

        target = (double)(rng_next(rng) >> 8) / (double)(1u << 24);
        target *= total;

        chosen = cand[k - 1].i;
        {
            double acc = 0.0;
            for (i = 0; i < k; ++i) {
                acc += p[i];
                if (acc >= target) { chosen = cand[i].i; break; }
            }
        }
        free(p);
    }

    free(cand);
    return chosen;
}

/* ------------------------------------------------------------------ */
/* Generation                                                          */
/* ------------------------------------------------------------------ */
int nita_generate(nita_model_t *m, const int32_t *prompt, size_t n_prompt,
                  int max_tokens, float temperature, int top_k, int32_t eos_id,
                  int32_t **out_ids, size_t *out_count, uint32_t seed,
                  bool *hit_limit)
{
    float   *logits;
    int32_t *ids = NULL;
    size_t   produced = 0;
    size_t   i;
    uint32_t rng = seed ? seed : 0xC0FFEEu;

    *out_ids = NULL;
    *out_count = 0;
    if (hit_limit) *hit_limit = false;

    if (!m || !m->ready) return 0;
    if (n_prompt == 0) return 0;
    if (max_tokens <= 0) return 0;

    nita_model_reset(m);

    /* Keep the most recent context if the prompt is too long. */
    if (n_prompt > (size_t)m->ctx) {
        prompt   += n_prompt - (size_t)m->ctx;
        n_prompt  = (size_t)m->ctx;
    }

    logits = (float *)nita_malloc((size_t)m->n_vocab * sizeof(float));
    ids    = (int32_t *)nita_malloc((size_t)max_tokens * sizeof(int32_t));

    /* Prefill */
    for (i = 0; i < n_prompt; ++i) {
        if (nita_forward(m, prompt[i], logits) != 0) {
            free(logits); free(ids);
            return 0;
        }
    }

    for (i = 0; i < (size_t)max_tokens; ++i) {
        int32_t next = nita_sample(logits, m->n_vocab, temperature, top_k, &rng);

        if (eos_id >= 0 && next == eos_id) break;

        ids[produced++] = next;

        /* Stop cleanly at the context boundary; report it as a length stop. */
        if (m->pos >= m->ctx) {
            if (hit_limit) *hit_limit = true;
            break;
        }
        if (nita_forward(m, next, logits) != 0) break;

        if ((size_t)(i + 1) >= (size_t)max_tokens) {
            if (hit_limit) *hit_limit = true;
        }
    }

    free(logits);

    if (produced == 0) {
        free(ids);
        return 0;
    }

    *out_ids = ids;
    *out_count = produced;
    return (int)produced;
}
