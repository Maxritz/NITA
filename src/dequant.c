/*
 * dequant.c – GGUF block dequantisation.
 *
 * The engine never materialises a whole dequantised weight matrix. Rows are
 * dequantised on demand into a small reusable buffer, which honours the
 * project rule of only expanding the tile that is actually being consumed.
 *
 * Block layouts follow the ggml definitions. Little-endian throughout.
 */

#include "nita.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Half precision                                                      */
/* ------------------------------------------------------------------ */
float nita_f16_to_f32(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    {
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
}

static uint16_t rd_u16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static float rd_f16(const uint8_t *p)
{
    return nita_f16_to_f32(rd_u16(p));
}

static float rd_f32(const uint8_t *p)
{
    uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
                  | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* ------------------------------------------------------------------ */
/* Byte size of one row                                                */
/* ------------------------------------------------------------------ */
size_t nita_row_bytes(uint32_t type, uint64_t ne0)
{
    uint64_t bs = ggml_blck_size(type);
    uint64_t sz = ggml_type_size(type);
    uint64_t blocks;

    if (bs == 0) bs = 1;
    blocks = (ne0 + bs - 1) / bs;
    return (size_t)(blocks * sz);
}

/* ------------------------------------------------------------------ */
/* K-quant scale unpacking (shared by Q4_K / Q5_K)                     */
/* ------------------------------------------------------------------ */
static void unpack_q4k_scales(const uint8_t *scales_in, int8_t scales[8], int8_t mins[8])
{
    uint32_t aux[4];
    uint32_t tmp;
    const uint32_t kmask1 = 0x3f3f3f3fu;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    int i;

    for (i = 0; i < 3; ++i) {
        aux[i] = (uint32_t)scales_in[i * 4 + 0]
               | ((uint32_t)scales_in[i * 4 + 1] << 8)
               | ((uint32_t)scales_in[i * 4 + 2] << 16)
               | ((uint32_t)scales_in[i * 4 + 3] << 24);
    }

    tmp   = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    for (i = 0; i < 8; ++i) {
        scales[i] = (int8_t)((aux[i / 4] >> (8 * (i % 4))) & 0xFF);
    }
    for (i = 0; i < 8; ++i) {
        mins[i] = (int8_t)((aux[2 + i / 4] >> (8 * (i % 4))) & 0xFF);
    }
}

/* ------------------------------------------------------------------ */
/* Row dequantisation                                                  */
/* ------------------------------------------------------------------ */
int nita_dequant_row(const uint8_t *src, uint32_t type, uint64_t ne0, float *dst)
{
    uint64_t i;

    switch (type) {
    /* ---------------- unquantised ---------------- */
    case 0: /* F32 */
        for (i = 0; i < ne0; ++i) dst[i] = rd_f32(src + i * 4);
        return 0;

    case 1: /* F16 */
        for (i = 0; i < ne0; ++i) dst[i] = rd_f16(src + i * 2);
        return 0;

    case 30: /* BF16: high 16 bits of an f32 */
        for (i = 0; i < ne0; ++i) {
            uint32_t bits = ((uint32_t)rd_u16(src + i * 2)) << 16;
            float f;
            memcpy(&f, &bits, 4);
            dst[i] = f;
        }
        return 0;

    case 28: /* F64 */
        for (i = 0; i < ne0; ++i) {
            uint64_t bits = 0;
            double d;
            int k;
            for (k = 7; k >= 0; --k) bits = (bits << 8) | src[i * 8 + (uint64_t)k];
            memcpy(&d, &bits, 8);
            dst[i] = (float)d;
        }
        return 0;

    case 24: /* I8 */
        for (i = 0; i < ne0; ++i) dst[i] = (float)(int8_t)src[i];
        return 0;

    case 25: /* I16 */
        for (i = 0; i < ne0; ++i) dst[i] = (float)(int16_t)rd_u16(src + i * 2);
        return 0;

    case 26: /* I32 */
        for (i = 0; i < ne0; ++i) {
            uint32_t v = (uint32_t)src[i * 4] | ((uint32_t)src[i * 4 + 1] << 8)
                       | ((uint32_t)src[i * 4 + 2] << 16) | ((uint32_t)src[i * 4 + 3] << 24);
            dst[i] = (float)(int32_t)v;
        }
        return 0;

    /* ---------------- legacy quants ---------------- */
    case 2: { /* Q4_0: 32 elements, 18 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 18;
            float d = rd_f16(blk);
            const uint8_t *qs = blk + 2;
            int j;
            for (j = 0; j < 16; ++j) {
                int lo = (qs[j] & 0x0F) - 8;
                int hi = (qs[j] >> 4) - 8;
                dst[i * 32 + j]      = d * (float)lo;
                dst[i * 32 + j + 16] = d * (float)hi;
            }
        }
        return 0;
    }

    case 3: { /* Q4_1: 32 elements, 20 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 20;
            float d = rd_f16(blk);
            float m = rd_f16(blk + 2);
            const uint8_t *qs = blk + 4;
            int j;
            for (j = 0; j < 16; ++j) {
                int lo = (qs[j] & 0x0F);
                int hi = (qs[j] >> 4);
                dst[i * 32 + j]      = d * (float)lo + m;
                dst[i * 32 + j + 16] = d * (float)hi + m;
            }
        }
        return 0;
    }

    case 6: { /* Q5_0: 32 elements, 22 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 22;
            float d = rd_f16(blk);
            uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8)
                        | ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
            const uint8_t *qs = blk + 6;
            int j;
            for (j = 0; j < 16; ++j) {
                int xh0 = (int)((qh >> j) & 1) << 4;
                int xh1 = (int)((qh >> (j + 16)) & 1) << 4;
                int lo = (qs[j] & 0x0F) | xh0;
                int hi = (qs[j] >> 4) | xh1;
                dst[i * 32 + j]      = d * (float)(lo - 16);
                dst[i * 32 + j + 16] = d * (float)(hi - 16);
            }
        }
        return 0;
    }

    case 7: { /* Q5_1: 32 elements, 24 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 24;
            float d = rd_f16(blk);
            float m = rd_f16(blk + 2);
            uint32_t qh = (uint32_t)blk[4] | ((uint32_t)blk[5] << 8)
                        | ((uint32_t)blk[6] << 16) | ((uint32_t)blk[7] << 24);
            const uint8_t *qs = blk + 8;
            int j;
            for (j = 0; j < 16; ++j) {
                int xh0 = (int)((qh >> j) & 1) << 4;
                int xh1 = (int)((qh >> (j + 16)) & 1) << 4;
                int lo = (qs[j] & 0x0F) | xh0;
                int hi = (qs[j] >> 4) | xh1;
                dst[i * 32 + j]      = d * (float)lo + m;
                dst[i * 32 + j + 16] = d * (float)hi + m;
            }
        }
        return 0;
    }

    case 8: { /* Q8_0: 32 elements, 34 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 34;
            float d = rd_f16(blk);
            const int8_t *qs = (const int8_t *)(blk + 2);
            int j;
            for (j = 0; j < 32; ++j)
                dst[i * 32 + j] = d * (float)qs[j];
        }
        return 0;
    }

    case 9: { /* Q8_1: 32 elements, 36 bytes */
        uint64_t nb = ne0 / 32;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 36;
            float d = rd_f16(blk);
            const int8_t *qs = (const int8_t *)(blk + 4);
            int j;
            for (j = 0; j < 32; ++j)
                dst[i * 32 + j] = d * (float)qs[j];
        }
        return 0;
    }

    /* ---------------- k-quants (super-block = 256) ---------------- */
    case 10: { /* Q2_K: 256 elements, 84 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 84;
            const uint8_t *scales = blk;
            const uint8_t *qs = blk + 16;
            float d    = rd_f16(blk + 80);
            float dmin = rd_f16(blk + 82);
            int is = 0;
            int n;
            float *y = dst + i * 256;

            for (n = 0; n < 256; n += 128) {
                int shift = 0;
                int j;
                for (j = 0; j < 4; ++j) {
                    uint8_t sc = scales[is++];
                    float dl = d * (float)(sc & 0x0F);
                    float ml = dmin * (float)(sc >> 4);
                    int l;
                    for (l = 0; l < 16; ++l)
                        y[n + j * 32 + l] = dl * (float)((qs[l] >> shift) & 3) - ml;

                    sc = scales[is++];
                    dl = d * (float)(sc & 0x0F);
                    ml = dmin * (float)(sc >> 4);
                    for (l = 0; l < 16; ++l)
                        y[n + j * 32 + 16 + l] = dl * (float)((qs[l + 16] >> shift) & 3) - ml;
                    shift += 2;
                }
                qs += 32;
            }
        }
        return 0;
    }

    case 11: { /* Q3_K: 256 elements, 110 bytes */
        uint64_t nb = ne0 / 256;
        static const uint32_t kmask1 = 0x03030303u;
        static const uint32_t kmask2 = 0x0f0f0f0fu;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 110;
            const uint8_t *hmask = blk;
            const uint8_t *qs = blk + 32;
            const uint8_t *scale_bytes = blk + 96;
            float d_all = rd_f16(blk + 108);
            uint32_t aux[4];
            int8_t scales[16];
            int is = 0;
            int n;
            float *y = dst + i * 256;
            uint8_t m = 1;

            for (is = 0; is < 3; ++is) {
                aux[is] = (uint32_t)scale_bytes[is * 4 + 0]
                        | ((uint32_t)scale_bytes[is * 4 + 1] << 8)
                        | ((uint32_t)scale_bytes[is * 4 + 2] << 16)
                        | ((uint32_t)scale_bytes[is * 4 + 3] << 24);
            }
            {
                uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = ((aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4));
                aux[1] = ((aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4));
            }
            for (is = 0; is < 16; ++is) {
                int v = (int)((aux[is / 4] >> (8 * (is % 4))) & 0xFF);
                scales[is] = (int8_t)v;
            }

            is = 0;
            for (n = 0; n < 256; n += 128) {
                int shift = 0;
                int j;
                for (j = 0; j < 4; ++j) {
                    float dl = d_all * (float)(scales[is++] - 32);
                    int l;
                    for (l = 0; l < 16; ++l) {
                        int q = ((qs[l] >> shift) & 3)
                              - ((hmask[l] & m) ? 0 : 4);
                        y[n + j * 32 + l] = dl * (float)q;
                    }
                    dl = d_all * (float)(scales[is++] - 32);
                    for (l = 0; l < 16; ++l) {
                        int q = ((qs[l + 16] >> shift) & 3)
                              - ((hmask[l + 16] & m) ? 0 : 4);
                        y[n + j * 32 + 16 + l] = dl * (float)q;
                    }
                    shift += 2;
                    m <<= 1;
                }
                qs += 32;
            }
            (void)hmask;
        }
        return 0;
    }

    case 12: { /* Q4_K: 256 elements, 144 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 144;
            float d    = rd_f16(blk);
            float dmin = rd_f16(blk + 2);
            int8_t scales[8], mins[8];
            const uint8_t *qs = blk + 16;
            float *y = dst + i * 256;
            int j;

            unpack_q4k_scales(blk + 4, scales, mins);

            for (j = 0; j < 8; ++j) {
                float d1 = d * (float)scales[j];
                float m1 = dmin * (float)mins[j];
                int l;
                for (l = 0; l < 16; ++l)
                    y[j * 32 + l] = d1 * (float)(qs[l] & 0x0F) - m1;
                for (l = 0; l < 16; ++l)
                    y[j * 32 + 16 + l] = d1 * (float)(qs[l] >> 4) - m1;
                qs += 16;
            }
        }
        return 0;
    }

    case 13: { /* Q5_K: 256 elements, 176 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 176;
            float d    = rd_f16(blk);
            float dmin = rd_f16(blk + 2);
            int8_t scales[8], mins[8];
            const uint8_t *qh = blk + 16;
            const uint8_t *ql = blk + 48;
            float *y = dst + i * 256;
            uint8_t u1 = 1, u2 = 2;
            int j;

            unpack_q4k_scales(blk + 4, scales, mins);

            for (j = 0; j < 4; ++j) {
                float d1 = d * (float)scales[2 * j + 0];
                float m1 = dmin * (float)mins[2 * j + 0];
                float d2 = d * (float)scales[2 * j + 1];
                float m2 = dmin * (float)mins[2 * j + 1];
                int l;
                for (l = 0; l < 32; ++l) {
                    int hi = (qh[l] & u1) ? 16 : 0;
                    y[j * 64 + l] = d1 * (float)((ql[l] & 0x0F) + hi) - m1;
                }
                for (l = 0; l < 32; ++l) {
                    int hi = (qh[l] & u2) ? 16 : 0;
                    y[j * 64 + 32 + l] = d2 * (float)((ql[l] >> 4) + hi) - m2;
                }
                ql += 32;
                u1 = (uint8_t)(u1 << 2);
                u2 = (uint8_t)(u2 << 2);
            }
        }
        return 0;
    }

    case 14: { /* Q6_K: 256 elements, 210 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 210;
            const uint8_t *ql = blk;
            const uint8_t *qh = blk + 128;
            const int8_t  *sc = (const int8_t *)(blk + 192);
            float d = rd_f16(blk + 208);
            float *y = dst + i * 256;
            int n;

            for (n = 0; n < 256; n += 128) {
                int l;
                for (l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    y[n + l + 0]  = d * (float)sc[is + 0] * (float)q1;
                    y[n + l + 32] = d * (float)sc[is + 2] * (float)q2;
                    y[n + l + 64] = d * (float)sc[is + 4] * (float)q3;
                    y[n + l + 96] = d * (float)sc[is + 6] * (float)q4;
                }
                /* y must not advance here: n already carries the 128-element
                 * half, and doing both walked 256 floats past the row. */
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return 0;
    }

    case 23: { /* IQ4_XS: 256 elements, 136 bytes */
        /* Non-linear 4-bit codebook; the values are the ggml table verbatim.
         * Indexing it with the nibble is the whole of IQ4_XS decoding: unlike
         * Q4_K the nibble is not a linear offset. */
        static const int8_t kv[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10,
               1,   13,  25,  38,  53,  69,  89, 113
        };
        uint64_t nb = ne0 / 256;

        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 136;
            float          d   = rd_f16(blk);
            uint16_t       h   = (uint16_t)(blk[2] | (blk[3] << 8));
            const uint8_t *sl  = blk + 4;
            const uint8_t *qs  = blk + 8;
            float         *y   = dst + i * 256;
            int            ib;

            for (ib = 0; ib < 8; ++ib) {
                int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0x0F)
                       | (((h >> (2 * ib)) & 3) << 4);
                float dl = d * (float)(ls - 32);
                int   l;

                for (l = 0; l < 16; ++l) {
                    y[ib * 32 + l]      = dl * (float)kv[qs[ib * 16 + l] & 0x0F];
                    y[ib * 32 + 16 + l] = dl * (float)kv[qs[ib * 16 + l] >> 4];
                }
            }
        }
        return 0;
    }

    case 39: { /* MXFP4: 32 elements, 17 bytes: E8M0 exponent + 16 code bytes */
        /* e2m1 values, doubled, so the scale carries the missing half. */
        static const int8_t kv[16] = {
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
        };
        uint64_t nb = ne0 / 32;

        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 17;
            float          d   = exp2f((float)blk[0] - 128.0f);
            int            l;

            for (l = 0; l < 16; ++l) {
                dst[i * 32 + l]      = d * (float)kv[blk[1 + l] & 0x0F];
                dst[i * 32 + 16 + l] = d * (float)kv[blk[1 + l] >> 4];
            }
        }
        return 0;
    }

    case 15: { /* Q8_K: 256 elements, 292 bytes */
        uint64_t nb = ne0 / 256;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 292;
            float d = rd_f32(blk);
            const int8_t *qs = (const int8_t *)(blk + 4);
            int j;
            for (j = 0; j < 256; ++j)
                dst[i * 256 + j] = d * (float)qs[j];
        }
        return 0;
    }

    /* ---- binary / ternary formats (PrismML Bonsai family) ---------- */

    case 41: { /* Q1_0: 128 elements, 18 bytes, one sign bit per weight */
        uint64_t nb = ne0 / 128;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 18;
            float d = rd_f16(blk);
            const uint8_t *qs = blk + 2;
            int j;
            for (j = 0; j < 128; ++j) {
                int bit = (qs[j / 8] >> (j % 8)) & 1;
                dst[i * 128 + j] = bit ? d : -d;
            }
        }
        return 0;
    }

    case 42: { /* Q2_0: 64 elements, 18 bytes, ternary in 2-bit slots */
        uint64_t nb = ne0 / 64;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 18;
            float d = rd_f16(blk);
            int j;
            for (j = 0; j < 64; ++j) {
                int q = (blk[2 + j / 4] >> ((j % 4) * 2)) & 3;
                dst[i * 64 + j] = (float)(q - 1) * d;
            }
        }
        return 0;
    }

    case 142: { /* PQ2_0: 128 elements, 34 bytes, ternary group 128 */
        uint64_t nb = ne0 / 128;
        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 34;
            float d = rd_f16(blk);
            int j;
            for (j = 0; j < 128; ++j) {
                int q = (blk[2 + j / 4] >> ((j % 4) * 2)) & 3;
                dst[i * 128 + j] = (float)(q - 1) * d;
            }
        }
        return 0;
    }

    case 143: { /* PTQ1_0: 128 elements, 28 bytes, dense base-3 trits */
        static const size_t  stages[3] = { 32, 16, 8 };
        static const uint8_t pow3[6]    = { 1, 3, 9, 27, 81, 243 };
        uint64_t nb = ne0 / 128;

        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 28;
            const uint8_t *qs  = blk;        /* 24 bytes: 5 trits each */
            const uint8_t *qh  = blk + 24;   /*  2 bytes: 4 trits each */
            float d = rd_f16(blk + 26);
            float *y = dst + i * 128;
            size_t j = 0, s, m, n;

            for (s = 0; s < 3; ++s) {
                size_t c = stages[s];
                for (; j + c <= 24; j += c) {
                    for (n = 0; n < 5; ++n) {
                        for (m = 0; m < c; ++m) {
                            /* truncating the product to 8 bits then ceiling-
                             * dividing by 243 isolates the n-th trit */
                            uint8_t q  = (uint8_t)(qs[j + m] * pow3[n]);
                            int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                            *y++ = (float)(xi - 1) * d;
                        }
                    }
                }
            }
            for (n = 0; n < 4; ++n) {
                for (m = 0; m < 2; ++m) {
                    uint8_t q  = (uint8_t)(qh[m] * pow3[n]);
                    int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                    *y++ = (float)(xi - 1) * d;
                }
            }
        }
        return 0;
    }

    case 34: { /* TQ1_0: 256 elements, 54 bytes, dense base-3 trits, group 256 */
        static const size_t  stages[3] = { 32, 16, 8 };
        static const uint8_t pow3[6]    = { 1, 3, 9, 27, 81, 243 };
        uint64_t nb = ne0 / 256;

        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 54;
            const uint8_t *qs  = blk;        /* 48 bytes: 5 trits each */
            const uint8_t *qh  = blk + 48;   /*  4 bytes: 4 trits each */
            float d = rd_f16(blk + 52);
            float *y = dst + i * 256;
            size_t j = 0, s, m, n;

            /* the same staged walk PTQ1_0 uses: 32 bytes (160 values) then
             * 16 bytes (80 values), which is exactly upstream's hard-coded
             * 32-then-16 layout for a 48-byte qs */
            for (s = 0; s < 3; ++s) {
                size_t c = stages[s];
                for (; j + c <= 48; j += c) {
                    for (n = 0; n < 5; ++n) {
                        for (m = 0; m < c; ++m) {
                            uint8_t q  = (uint8_t)(qs[j + m] * pow3[n]);
                            int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                            *y++ = (float)(xi - 1) * d;
                        }
                    }
                }
            }
            for (n = 0; n < 4; ++n) {
                for (m = 0; m < 4; ++m) {
                    uint8_t q  = (uint8_t)(qh[m] * pow3[n]);
                    int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
                    *y++ = (float)(xi - 1) * d;
                }
            }
        }
        return 0;
    }

    case 35: { /* TQ2_0: 256 elements, 66 bytes, 2-bit slots, group 256 */
        uint64_t nb = ne0 / 256;

        for (i = 0; i < nb; ++i) {
            const uint8_t *blk = src + i * 66;
            const uint8_t *qs  = blk;        /* 64 bytes, 4 slots each */
            float d = rd_f16(blk + 64);      /* scale last, unlike Q2_0 */
            float *y = dst + i * 256;
            size_t g, l, m;

            /* slot l is a bit plane across 32 stored bytes, so it covers 32
             * consecutive elements: upstream calls this "along 32 bytes" */
            for (g = 0; g < 2; ++g)
                for (l = 0; l < 4; ++l)
                    for (m = 0; m < 32; ++m)
                        *y++ = (float)(((qs[g * 32 + m] >> (2 * l)) & 3) - 1) * d;
        }
        return 0;
    }

    default:
        return -1;
    }
}
