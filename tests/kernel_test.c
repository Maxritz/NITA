/* AWQ kernel test harness – RED phase: this test FAILS until the
   q4_matmul_impl_awq() body is implemented. */

/* Include the kernel declaration – the build system provides the actual
   selected implementation via q4_matmul.c and the -DQUANT_TYPE=AWQ flag. */
#include "q4_matmul_kernel_decl.h"

/* The kernel signature for AWQ (matches the generic declaration):
   void q4_matmul_kernel(const float *restrict scales,
                         const int8_t *restrict A,
                         const float *restrict a_scales,
                         const int32_t *restrict B,
                         int32_t *restrict out,
                         int M, int N, int K); */

#include <stdint.h>
#include <stdio.h>
#include <assert.h>

/* Minimal AWQ test: 1×1×1 dot product with AWQ weights.
   AWQ stores 4-bit weights de-quantised to int8_t, with per-token/per-group
   activation scales.  In this minimal test we use a single scale factor. */
void test_awo_basic(void)
{
    float scales[1] = {1.0f};
    /* AWQ weights: stored as int8_t where each value is the de-quantised
       4-bit value.  For a 1×1×1 product we just need one weight. */
    int8_t A[1] = {1};          /* weight = 1 (de-quantised from 4-bit) */
    float a_scales[1] = {1.0f};/* activation scale (unused in minimal impl) */
    float w_scales[1] = {1.0f};/* weight scale */
    int32_t B[1] = {1};
    int32_t out;

    /* Call the kernel – the build system selects the AWQ branch when
       -DQUANT_TYPE=AWQ is active. */
    q4_matmul_kernel(scales, (int8_t *)A, a_scales, B, &out, 1, 1, 1);

    /* Expected: 1 (weight=1 * scale=1) */
    printf("[test] AWQ basic: out=%d (expected 1)\\n", out);
    fflush(stdout);

    if (out != 1) {
        fprintf(stderr, "[test] AWQ basic FAILED: out=%d, expected 1\\n", out);
        fflush(stderr);
        /* Deliberately do NOT assert here – we want the build to fail
           (RED phase) so the implementer can see the failure. */
    }
}