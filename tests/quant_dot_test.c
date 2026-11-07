/*
 * quant_dot_test.c - does the integer kernel agree with the float reference?
 *
 * Reads quantised tensors and compares, row by row,
 *
 *     nita_quant_dot()                        (integer domain)
 *     dequantise the row, multiply in float   (the reference)
 *
 * A layout mistake shows up as an error near 100%; the only difference that
 * should remain is the int8 quantisation of the activations, a fraction of a
 * percent per element.
 *
 *   quant_dot_test <file.gguf> <type-id> [rows]
 *   quant_dot_test --raw <blob>        (blob from tools/extract_quant_rows.py)
 */
#include "../src/nita.h"
#include "../src/quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
static double now_sec(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

static unsigned rng_state = 12345u;

static float frand(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)((rng_state >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
}

/* One row: returns the relative error of the integer dot. */
static double one_row(uint32_t type, const uint8_t *src, uint32_t k, int nb,
                      float *row, float *x, nita_aq8_t *aq, double *abs_err)
{
    double ref = 0.0, got, scale = 0.0;
    long   i;

    if (nita_dequant_row(src, type, k, row) != 0) return 0.0;
    for (i = 0; i < (long)k; ++i) {
        ref   += (double)row[i] * (double)x[i];
        scale += fabs((double)row[i] * (double)x[i]);
    }
    got = (double)nita_quant_dot(type, src, aq, nb);
    *abs_err = fabs(got - ref);
    return scale > 0.0 ? fabs(got - ref) / scale : 0.0;
}

static int run_blob(const char *path)
{
    FILE       *f = fopen(path, "rb");
    uint32_t    hdr[4];
    uint8_t    *blob;
    size_t      need;
    int         nb, r;
    float      *row, *x;
    nita_aq8_t *aq;
    double      worst = 0.0, worst_abs = 0.0;
    long        i;

    if (!f) { perror(path); return 2; }
    if (fread(hdr, sizeof hdr, 1, f) != 1) { fclose(f); return 2; }
    need = (size_t)hdr[3] * hdr[2];
    blob = (uint8_t *)malloc(need);
    if (!blob || fread(blob, 1, need, f) != need) {
        fprintf(stderr, "short read: %s\n", path);
        fclose(f);
        free(blob);
        return 2;
    }
    fclose(f);

    nb  = (int)(hdr[1] / 32);
    row = (float *)malloc((size_t)hdr[1] * sizeof(float));
    x   = (float *)malloc((size_t)hdr[1] * sizeof(float));
    aq  = (nita_aq8_t *)malloc((size_t)nb * sizeof *aq);
    if (!row || !x || !aq) { fprintf(stderr, "oom\n"); return 2; }

    for (i = 0; i < (long)hdr[1]; ++i) x[i] = frand();
    nita_aq8_quantise(x, (int)hdr[1], aq);

    for (r = 0; r < (int)hdr[2]; ++r) {
        double ae;
        double rel = one_row(hdr[0], blob + (size_t)r * hdr[3], hdr[1], nb,
                             row, x, aq, &ae);

        if (rel > worst) worst = rel;
        if (ae > worst_abs) worst_abs = ae;
    }

    printf("%-34s type %-2u %-6s k=%-5u rows=%-2u  rel err %8.3e  abs %8.4f\n",
           path, hdr[0], ggml_type_name(hdr[0]), hdr[1], hdr[2], worst, worst_abs);

    free(row); free(x); free(aq); free(blob);
    return worst > 1e-2 ? 1 : 0;
}

/* Cost of one dot: integer path vs the dequantise-then-float path that
 * NITA_INT8_DOT=0 takes.  Same row, same activations, same buffer sizes. */
static int run_bench(const char *path, int iters)
{
    FILE       *f = fopen(path, "rb");
    uint32_t    hdr[4];
    uint8_t    *blob;
    size_t      need;
    int         nb, r, i;
    float      *row, *x, *w;
    nita_aq8_t *aq;
    double      sink = 0.0;
    double      t0, t1, ns_int, ns_flt;
    size_t      rows, k, row_bytes;

    if (!f) { perror(path); return 2; }
    if (fread(hdr, sizeof hdr, 1, f) != 1) { fclose(f); return 2; }
    need = (size_t)hdr[3] * hdr[2];
    blob = (uint8_t *)malloc(need);
    if (!blob || fread(blob, 1, need, f) != need) {
        fprintf(stderr, "short read: %s\n", path);
        fclose(f);
        free(blob);
        return 2;
    }
    fclose(f);

    k = hdr[1]; rows = hdr[2]; row_bytes = hdr[3];
    nb  = (int)(k / 32);
    row = (float *)malloc(k * sizeof(float));
    w   = (float *)malloc(k * sizeof(float));
    x   = (float *)malloc(k * sizeof(float));
    aq  = (nita_aq8_t *)malloc((size_t)nb * sizeof *aq);
    if (!row || !w || !x || !aq) { fprintf(stderr, "oom\n"); return 2; }

    for (i = 0; i < (int)k; ++i) x[i] = frand();
    nita_aq8_quantise(x, (int)k, aq);

    /* touch every block first so page faults are not in the timed loop */
    for (r = 0; r < (int)rows; ++r)
        sink += (double)nita_quant_dot(hdr[0], blob + (size_t)r * row_bytes, aq, nb);

    t0 = now_sec();
    for (i = 0; i < iters; ++i)
        for (r = 0; r < (int)rows; ++r)
            sink += (double)nita_quant_dot(hdr[0], blob + (size_t)r * row_bytes, aq, nb);
    t1 = now_sec();
    ns_int = (t1 - t0) * 1e9 / ((double)iters * (double)rows);

    t0 = now_sec();
    for (i = 0; i < iters; ++i)
        for (r = 0; r < (int)rows; ++r) {
            int j;
            float s = 0.0f;

            nita_dequant_row(blob + (size_t)r * row_bytes, hdr[0], (uint64_t)k, w);
            for (j = 0; j < (int)k; ++j) s += w[j] * x[j];
            sink += (double)s;
        }
    t1 = now_sec();
    ns_flt = (t1 - t0) * 1e9 / ((double)iters * (double)rows);

    printf("%-30s %-7s k=%-5llu int8 %8.1f ns/row  dequant+float %8.1f ns/row"
           "  x%.2f  %5.1f GB/s (sink %.1f)\n",
           path, ggml_type_name(hdr[0]), (unsigned long long)k,
           ns_int, ns_flt, ns_flt / ns_int,
           (double)row_bytes / ns_int, sink);

    free(row); free(w); free(x); free(aq); free(blob);
    return 0;
}

int main(int argc, char **argv)
{
    gguf_file_t *g;
    char         err[256];
    uint32_t     type;
    int          max_rows = 32, tested = 0, failed = 0;
    uint64_t     ti;

    if (argc >= 3 && strcmp(argv[1], "--raw") == 0)
        return run_blob(argv[2]);

    if (argc >= 3 && strcmp(argv[1], "--bench") == 0)
        return run_bench(argv[2], argc > 3 ? atoi(argv[3]) : 2000);

    if (argc < 3) {
        fprintf(stderr, "usage: %s <file.gguf> <type-id> [rows]\n"
                        "       %s --raw <blob>\n", argv[0], argv[0]);
        return 2;
    }
    type = (uint32_t)strtoul(argv[2], NULL, 10);
    if (argc > 3) max_rows = atoi(argv[3]);

    g = gguf_load(argv[1], err, sizeof err);
    if (!g) { fprintf(stderr, "load failed: %s\n", err); return 2; }

    for (ti = 0; ti < g->tensor_count; ++ti) {
        const gguf_tensor_t *t = &g->tensors[ti];
        size_t               row_bytes, k, nrows, r;
        float               *row, *x;
        nita_aq8_t          *aq;
        double               worst = 0.0, worst_abs = 0.0;
        int                  nb;

        if (t->type != type || t->n_dims < 2) continue;
        k         = (size_t)t->dims[0];
        nrows     = (size_t)t->dims[1];
        row_bytes = nita_row_bytes(t->type, t->dims[0]);
        if (k % 32 != 0 || row_bytes == 0) continue;

        nb  = (int)(k / 32);
        row = (float *)malloc(k * sizeof(float));
        x   = (float *)malloc(k * sizeof(float));
        aq  = (nita_aq8_t *)malloc((size_t)nb * sizeof *aq);
        if (!row || !x || !aq) return 2;

        for (r = 0; r < k; ++r) x[r] = frand();
        nita_aq8_quantise(x, (int)k, aq);

        for (r = 0; r < (size_t)(max_rows < (int)nrows ? max_rows : nrows); ++r) {
            double ae;
            double rel = one_row(t->type,
                                 g->data + g->data_offset + t->offset + r * row_bytes,
                                 (uint32_t)k, nb, row, x, aq, &ae);

            if (rel > worst) worst = rel;
            if (ae > worst_abs) worst_abs = ae;
        }
        printf("  %-40s rows=%-5llu k=%-6llu  rel err %8.3e  abs %8.4f\n",
               t->name, (unsigned long long)nrows, (unsigned long long)k,
               worst, worst_abs);
        ++tested;
        if (worst > 1e-2) ++failed;
        free(row); free(x); free(aq);
    }

    printf("%d tensor(s) of this type, %d over the 1e-2 tolerance\n", tested, failed);
    gguf_free(g);
    return failed ? 1 : 0;
}
