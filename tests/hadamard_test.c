/*
 * hadamard_test.c - does the block FWHT reproduce the PrismML rotation, and is
 * it an orthogonal involution?
 *
 *   1. forward() must equal a direct multiply by the reference matrix
 *      R[row][col] = (popcount(row & col) & 1 ? -1 : +1) / sqrt(block), built
 *      here from nita_hadamard_matrix_row() the way the loader builds it.
 *   2. R must be orthogonal and symmetric, so forward twice is the identity and
 *      inner products are preserved.
 *   3. inverse(forward(x)) == x, including with an explicit sign vector.
 *   4. cost per 2048-element row.
 */
#include "../src/hadamard.h"

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

static unsigned rng = 12345u;
static float frand(void)
{
    rng = rng * 1664525u + 1013904223u;
    return ((float)((rng >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
}

static int failures;

static void check(int ok, const char *what, double got, double want)
{
    printf("%-52s %s (got %.3e, ref %.3e)\n", what, ok ? "ok" : "FAIL", got, want);
    if (!ok) ++failures;
}

/* forward() against a direct matrix multiply over one block */
static void test_matches_reference(int block)
{
    float *x    = (float *)malloc((size_t)block * sizeof *x);
    float *want = (float *)malloc((size_t)block * sizeof *want);
    float *row  = (float *)malloc((size_t)block * sizeof *row);
    double worst = 0.0;
    nita_hadamard_t h;
    char err[256];
    int  r, c;

    for (r = 0; r < block; ++r) x[r] = frand();
    if (nita_hadamard_init(&h, block, 0, NULL, err, sizeof err) != 0) {
        printf("init failed: %s\n", err);
        ++failures;
        free(x); free(want); free(row);
        return;
    }

    for (r = 0; r < block; ++r) {
        double s = 0.0;

        nita_hadamard_matrix_row(block, r, row);
        for (c = 0; c < block; ++c) s += (double)row[c] * (double)x[c];
        want[r] = (float)s;
    }

    nita_hadamard_forward(x, block, &h);
    for (r = 0; r < block; ++r) {
        const double d = fabs((double)x[r] - (double)want[r]);

        if (d > worst) worst = d;
    }
    {
        char what[96];

        snprintf(what, sizeof what, "block %d: FWHT matches reference matrix", block);
        check(worst < 1e-5, what, worst, 0.0);
    }
    nita_hadamard_free(&h);
    free(x); free(want); free(row);
}

/* forward twice == identity (R is an involution up to the scale, R^2 = I) */
static void test_involution(int block)
{
    float *x   = (float *)malloc((size_t)block * sizeof *x);
    float *ref = (float *)malloc((size_t)block * sizeof *ref);
    double worst = 0.0;
    nita_hadamard_t h;
    char err[256];
    int  i;

    for (i = 0; i < block; ++i) { x[i] = frand(); ref[i] = x[i]; }
    nita_hadamard_init(&h, block, 0, NULL, err, sizeof err);
    nita_hadamard_forward(x, block, &h);
    nita_hadamard_forward(x, block, &h);
    for (i = 0; i < block; ++i) {
        const double d = fabs((double)x[i] - (double)ref[i]);

        if (d > worst) worst = d;
    }
    {
        char what[96];

        snprintf(what, sizeof what, "block %d: forward twice is the identity", block);
        check(worst < 1e-5, what, worst, 0.0);
    }
    nita_hadamard_free(&h);
    free(x); free(ref);
}

/* <Ax, Ay> == <x, y>, and norm is preserved */
static void test_orthogonal(int n, int block, int width, int with_signs)
{
    float *x = (float *)malloc((size_t)n * sizeof *x);
    float *y = (float *)malloc((size_t)n * sizeof *y);
    float *signs = NULL;
    double dot_before = 0.0, dot_after = 0.0, worst;
    nita_hadamard_t h;
    char err[256];
    int  i;

    if (with_signs) {
        signs = (float *)malloc((size_t)width * sizeof *signs);
        for (i = 0; i < width; ++i) signs[i] = (i % 3 == 0) ? -1.0f : 1.0f;
    }
    for (i = 0; i < n; ++i) { x[i] = frand(); y[i] = frand(); }
    for (i = 0; i < n; ++i) dot_before += (double)x[i] * (double)y[i];

    if (nita_hadamard_init(&h, block, with_signs ? width : 0, signs, err, sizeof err) != 0) {
        printf("init failed: %s\n", err);
        ++failures;
        free(x); free(y); free(signs);
        return;
    }
    nita_hadamard_forward(x, n, &h);
    nita_hadamard_forward(y, n, &h);
    for (i = 0; i < n; ++i) dot_after += (double)x[i] * (double)y[i];

    worst = fabs(dot_after - dot_before);
    {
        char what[112];

        snprintf(what, sizeof what,
                 "n=%d block=%d%s: inner products preserved", n, block,
                 with_signs ? " with signs" : "");
        check(worst < 1e-4 * (1.0 + fabs(dot_before)), what, worst, 0.0);
    }
    nita_hadamard_free(&h);
    free(x); free(y); free(signs);
}

static void test_round_trip(int n, int block, int width)
{
    float *x   = (float *)malloc((size_t)n * sizeof *x);
    float *ref = (float *)malloc((size_t)n * sizeof *ref);
    float *signs = (float *)malloc((size_t)width * sizeof *signs);
    nita_hadamard_t h;
    char  err[256];
    double worst = 0.0;
    int   i;

    for (i = 0; i < width; ++i) signs[i] = (i % 5 == 0) ? -1.0f : 1.0f;
    for (i = 0; i < n; ++i) { x[i] = frand(); ref[i] = x[i]; }

    nita_hadamard_init(&h, block, width, signs, err, sizeof err);
    nita_hadamard_forward(x, n, &h);
    nita_hadamard_inverse(x, n, &h);
    for (i = 0; i < n; ++i) {
        const double d = fabs((double)x[i] - (double)ref[i]);

        if (d > worst) worst = d;
    }
    {
        char what[112];

        snprintf(what, sizeof what, "n=%d block=%d: inverse undoes forward", n, block);
        check(worst < 1e-5, what, worst, 0.0);
    }
    nita_hadamard_free(&h);
    free(x); free(ref); free(signs);
}

/* a sign vector must actually change the result, or the test above is vacuous */
static void test_signs_matter(int n, int block, int width)
{
    float *a = (float *)malloc((size_t)n * sizeof *a);
    float *b = (float *)malloc((size_t)n * sizeof *b);
    float *signs = (float *)malloc((size_t)width * sizeof *signs);
    nita_hadamard_t h;
    char  err[256];
    double worst = 0.0;
    int   i;

    for (i = 0; i < width; ++i) signs[i] = -1.0f;
    for (i = 0; i < n; ++i) { a[i] = frand(); b[i] = a[i]; }

    nita_hadamard_init(&h, block, width, signs, err, sizeof err);
    nita_hadamard_forward(a, n, &h);
    for (i = 0; i < n; ++i) {
        const double d = fabs((double)a[i] - (double)b[i]);

        if (d > worst) worst = d;
    }
    check(worst > 1e-3, "all-negative signs change the result", worst, 0.0);
    nita_hadamard_free(&h);
    free(a); free(b); free(signs);
}

static double bench(int iters)
{
    const int n = 2048, block = 512;
    float    *x = (float *)malloc((size_t)n * sizeof *x);
    nita_hadamard_t h;
    char      err[256];
    double    t0, t1;
    int       i;

    for (i = 0; i < n; ++i) x[i] = frand();
    nita_hadamard_init(&h, block, 0, NULL, err, sizeof err);

    nita_hadamard_forward(x, n, &h);          /* warm */
    t0 = now_sec();
    for (i = 0; i < iters; ++i) nita_hadamard_forward(x, n, &h);
    t1 = now_sec();

    nita_hadamard_free(&h);
    free(x);
    return (t1 - t0) * 1e9 / (double)iters;
}

int main(void)
{
    int blocks[] = { 2, 8, 64, 128, 512, 1024 };
    int i;

    for (i = 0; i < (int)(sizeof blocks / sizeof blocks[0]); ++i) {
        test_matches_reference(blocks[i]);
        test_involution(blocks[i]);
        test_orthogonal(blocks[i], blocks[i], 0, 0);
    }

    test_orthogonal(2048, 512, 2048, 0);
    test_orthogonal(2048, 512, 2048, 1);
    test_orthogonal(6144, 512, 6144, 1);
    test_round_trip(2048, 512, 2048);
    test_round_trip(6144, 512, 6144);
    test_signs_matter(2048, 512, 2048);

    printf("\ncost: %.1f ns per 2048-element forward (block 512, 4 blocks)\n",
           bench(20000));
    printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures,
           failures == 1 ? "" : "s");
    return failures != 0;
}
