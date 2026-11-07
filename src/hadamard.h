/*
 * hadamard.h - the PrismML activation rotation ("prism.hadamard.*").
 *
 * A Bonsai checkpoint stores some matmul weights folded with a fixed rotation,
 * so the activation has to be rotated before the matmul. The transform is the
 * normalised Sylvester Walsh-Hadamard matrix applied along the input axis, in
 * blocks of `block_size`:
 *
 *     R[row][col] = (popcount(row & col) & 1 ? -1 : +1) / sqrt(block_size)
 *
 * (that is how the PrismML loader builds its rotation tensor, so it is the
 * definition this engine follows), optionally preceded by an elementwise
 * multiply with a per-width explicit sign vector of +-1.
 *
 * The fold is  y = R (s * x)  for weights listed in prism.hadamard.weight_names,
 * and its inverse  x = s * (R y)  for prism.hadamard.inverse_weight_names.
 * R is symmetric and orthogonal, so R * x is exactly the natural-order fast
 * Walsh-Hadamard transform scaled by 1/sqrt(block_size), which is what
 * nita_hadamard applies: O(n log n) per block instead of O(n^2).
 */
#ifndef NITA_HADAMARD_H
#define NITA_HADAMARD_H

#include <stddef.h>

typedef struct {
    int    block;   /* elements per transform block (power of two) */
    int    width;   /* sign-vector length; 0 when the sign mode is identity */
    float *signs;   /* width entries, each +1 or -1 */
} nita_hadamard_t;

/* Validate the geometry and allocate the sign vector (signs may be NULL when
 * width is 0). Returns 0 on success, -1 with err filled on failure. */
int  nita_hadamard_init(nita_hadamard_t *h, int block, int width, const float *signs,
                        char *err, size_t errlen);
void nita_hadamard_free(nita_hadamard_t *h);

/* y = R (s * x) over n elements, n a multiple of block. */
void nita_hadamard_forward(float *x, int n, const nita_hadamard_t *h);

/* x = s * (R y) over n elements, the inverse of nita_hadamard_forward. */
void nita_hadamard_inverse(float *x, int n, const nita_hadamard_t *h);

/* One row of the reference matrix, for tests: out[0..block-1] = R[row][*]. */
void nita_hadamard_matrix_row(int block, int row, float *out);

/* Unnormalised in-place FWHT of n elements (n a power of two), exposed for
 * tests; nita_hadamard_forward divides by sqrt(block) around it. */
void nita_fwht(float *x, int n);

#endif
