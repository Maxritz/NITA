/*
 * q4_matmul_kernel_decl.h – minimal header for test compilation.
 * This file is NOT part of the normal build; it's only for the test harness.
 * The actual kernel selection is performed by CMake's -DQUANT_TYPE mechanism.
 */

#ifndef Q4_MATMUL_KERNEL_DECL_H
#define Q4_MATMUL_KERNEL_DECL_H

/* Generic AWQ kernel declaration – the build system will replace this
   with the appropriate #if QUANT_TYPE == AWQ branch from q4_matmul.c. */
#if defined(QUANT_TYPE_AWQ)
/* The real implementation lives in src/q4_matmul.c under #if QUANT_TYPE == AWQ.
   This header only provides a declaration so the test can compile. */
extern void q4_matmul_kernel(const float *restrict scales,
                              const int8_t *restrict A,
                              const float *restrict a_scales,
                              const int32_t *restrict B,
                              int32_t *restrict out,
                              int M, int N, int K);
#else
/* Fallback declaration for when AWQ is not the selected quant type. */
extern void q4_matmul_kernel(const float *restrict scales,
                              const int8_t *restrict A,
                              const float *restrict a_scales,
                              const int32_t *restrict B,
                              int32_t *restrict out,
                              int M, int N, int K);
#endif

#endif /* Q4_MATMUL_KERNEL_DECL_H */