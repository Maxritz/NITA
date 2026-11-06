/*
 * minimal_engine entry point – calls the quantised matmul kernel
 * and exits.  This file is required so the linker does not look for
 * WinMain (the default for C binaries on MinGW without a defined main).
 */

#include <stdint.h>
#include <stdio.h>

/* 
 * The kernel function signature varies based on the -D flags compiled in.
 * For a minimal test we just declare a generic version; the actual
 * selected implementation will match the CMake flags.
 * 
 * Signature (Q4_K_M default):
 *   void q4_matmul_kernel(const float *restrict scales,
 *                         const uint8_t *restrict A,
 *                         const float *restrict B_scales,
 *                         const int32_t *restrict B,
 *                         int32_t *restrict out,
 *                         int M, int N, int K);
 */

extern void q4_matmul_kernel(const float *restrict scales,
                              const uint8_t *restrict A,
                              const float *restrict B_scales,
                              const int32_t *restrict B,
                              int32_t *restrict out,
                              int M, int N, int K);

int main(void)
{
    /* Minimal exercise of the kernel – a 1×1×1 dot product with
       Q4_K_M (default).  The scales, A, B matrices are tiny stubs;
       the kernel will just accumulate and produce a result. */
    static float scales[1] = {1.0f};
    static uint8_t A[2] = {0x55, 0xAA};   /* two 4‑bit nibbles: 5 and 10 */
    static float B_scales[1] = {1.0f};
    static int32_t B[1] = {1};
    static int32_t out;

    /* Call the kernel – 1×1×1 GEMV. */
    q4_matmul_kernel(scales, A, B_scales, B, &out, 1, 1, 1);

    printf("[minimal_engine] Kernel executed, out=%d\\n", out);
    fflush(stdout);
    return 0;
}