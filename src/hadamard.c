/*
 * hadamard.c - block Sylvester Walsh-Hadamard rotation with explicit signs.
 *
 * See hadamard.h for the convention. The butterflies run in place, natural
 * (Sylvester) order, so the result equals R * x with
 * R[row][col] = (popcount(row & col) & 1 ? -1 : +1) / sqrt(n) - the matrix the
 * PrismML loader materialises - without ever building it.
 */

#include "hadamard.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
/*
 * The eight-point Walsh-Hadamard transform of eight consecutive elements,
 * held in one register.
 *
 * Strides 1, 2 and 4 are exactly the passes that cannot be vectorised as a
 * "load two halves" butterfly - each butterfly pair sits inside a lane - and
 * they touch every element, so done scalar they cost about eight times what
 * every later pass costs together. Doing them in-register turns three whole
 * passes into three shuffles, three add/sub pairs and three blends.
 */
static inline __m256 wht8(__m256 v)
{
    __m256 t, s, d;

    /* stride 1: pair lanes (0,1) (2,3) (4,5) (6,7) */
    t = _mm256_permute_ps(v, 0xB1);
    s = _mm256_add_ps(v, t);
    d = _mm256_sub_ps(v, t);
    v = _mm256_blend_ps(s, _mm256_permute_ps(d, 0xB1), 0xAA);

    /* stride 2 */
    t = _mm256_permute_ps(v, 0x4E);
    s = _mm256_add_ps(v, t);
    d = _mm256_sub_ps(v, t);
    v = _mm256_blend_ps(s, _mm256_permute_ps(d, 0x4E), 0xCC);

    /* stride 4: swap the 128-bit halves */
    t = _mm256_permute2f128_ps(v, v, 0x01);
    s = _mm256_add_ps(v, t);
    d = _mm256_sub_ps(v, t);
    v = _mm256_blend_ps(s, _mm256_permute2f128_ps(d, d, 0x01), 0xF0);

    return v;
}
#endif

void nita_fwht(float *x, int n)
{
    int len, i;

#if defined(__AVX2__)
    if (n >= 8) {
        for (i = 0; i < n; i += 8)
            _mm256_storeu_ps(x + i, wht8(_mm256_loadu_ps(x + i)));
        len = 8;
    } else {
        len = 1;
    }
#else
    len = 1;
#endif

    for (; len < n; len <<= 1) {
        for (i = 0; i < n; i += len << 1) {
            int j;

#if defined(__AVX2__)
            if (len >= 8) {
                for (j = i; j < i + len; j += 8) {
                    const __m256 a = _mm256_loadu_ps(x + j);
                    const __m256 b = _mm256_loadu_ps(x + j + len);

                    _mm256_storeu_ps(x + j, _mm256_add_ps(a, b));
                    _mm256_storeu_ps(x + j + len, _mm256_sub_ps(a, b));
                }
                continue;
            }
#endif
            for (j = i; j < i + len; ++j) {
                const float a = x[j];
                const float b = x[j + len];

                x[j]       = a + b;
                x[j + len] = a - b;
            }
        }
    }
}

/* scale every block by 1/sqrt(block), turning the unnormalised butterflies
 * into the normalised Sylvester matrix */
static void scale_blocks(float *x, int n, const nita_hadamard_t *h)
{
    const float scale = 1.0f / sqrtf((float)h->block);
    int         b;

#if defined(__AVX2__)
    const __m256 s = _mm256_set1_ps(scale);
#else
    int i;
#endif

    for (b = 0; b < n; b += h->block) {
#if defined(__AVX2__)
        int i2 = b;

        for (; i2 + 8 <= b + h->block; i2 += 8)
            _mm256_storeu_ps(x + i2, _mm256_mul_ps(_mm256_loadu_ps(x + i2), s));
        for (; i2 < b + h->block; ++i2) x[i2] *= scale;
#else
        for (i = b; i < b + h->block; ++i) x[i] *= scale;
#endif
    }
}

static void apply_signs(float *x, int n, const nita_hadamard_t *h)
{
    int i;

    if (!h->signs) return;

#if defined(__AVX2__)
    for (i = 0; i + 8 <= h->width && i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i,
                         _mm256_mul_ps(_mm256_loadu_ps(x + i),
                                       _mm256_loadu_ps(h->signs + i)));
    for (; i < h->width && i < n; ++i) x[i] *= h->signs[i];
#else
    for (i = 0; i < h->width && i < n; ++i) x[i] *= h->signs[i];
#endif
}

int nita_hadamard_init(nita_hadamard_t *h, int block, int width, const float *signs,
                       char *err, size_t errlen)
{
    int i;

    if (!h) return -1;
    memset(h, 0, sizeof *h);

    if (block <= 0 || (block & (block - 1)) != 0) {
        snprintf(err, errlen, "hadamard: block size %d is not a power of two", block);
        return -1;
    }
    if (width < 0 || width > block * 4096) {
        snprintf(err, errlen, "hadamard: implausible sign width %d", width);
        return -1;
    }

    h->block = block;
    h->width = width;
    if (width > 0 && signs) {
        h->signs = (float *)malloc((size_t)width * sizeof *h->signs);
        if (!h->signs) {
            snprintf(err, errlen, "hadamard: out of memory for %d signs", width);
            return -1;
        }
        for (i = 0; i < width; ++i) {
            const float s = signs[i];

            if (s != 1.0f && s != -1.0f) {
                snprintf(err, errlen,
                         "hadamard: sign %d is %g, expected +1 or -1", i, (double)s);
                free(h->signs);
                memset(h, 0, sizeof *h);
                return -1;
            }
            h->signs[i] = s;
        }
    } else {
        h->width = 0;
    }
    return 0;
}

void nita_hadamard_free(nita_hadamard_t *h)
{
    if (!h) return;
    free(h->signs);
    h->signs = NULL;
    h->width = 0;
}

void nita_hadamard_forward(float *x, int n, const nita_hadamard_t *h)
{
    int b;

    if (!h || h->block <= 0) return;

    /* y = R (s * x): signs, then the rotation */
    apply_signs(x, n, h);
    for (b = 0; b < n; b += h->block) nita_fwht(x + b, h->block);
    scale_blocks(x, n, h);
}

void nita_hadamard_inverse(float *x, int n, const nita_hadamard_t *h)
{
    int b;

    if (!h || h->block <= 0) return;

    /* x = s * (R y): R and the signs are both self-inverse, so the order flips */
    for (b = 0; b < n; b += h->block) nita_fwht(x + b, h->block);
    scale_blocks(x, n, h);
    apply_signs(x, n, h);
}

/* Reference: build one row directly from the parity definition of R. */
void nita_hadamard_matrix_row(int block, int row, float *out)
{
    const float scale = 1.0f / sqrtf((float)block);
    int         col;

    for (col = 0; col < block; ++col) {
        unsigned p = (unsigned)(row & col);

        p ^= p >> 16;
        p ^= p >> 8;
        p ^= p >> 4;
        p ^= p >> 2;
        p ^= p >> 1;
        out[col] = (p & 1u) ? -scale : scale;
    }
}
