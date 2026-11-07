/*
 * q35_heads_test.c - the value-head -> key-head mapping of the qwen35 mixer.
 *
 * Every value head of the gated delta-net owns a 128x128 state matrix but no
 * query of its own, so it has to borrow one of the k/q vectors. When the two
 * counts are equal that is trivially "head h", which is why the 2B never
 * exercised a choice. Bonsai 27B has 48 value heads and 16 key heads, and there
 * two readings are possible:
 *
 *   tiled   (correct) k = h % n_k_heads      -> 0 1 2 3 4 5 ...
 *   blocked (wrong)   k = h / (n_v/n_k)      -> 0 0 0 1 1 1 ...
 *
 * The reference runtime tiles: the fused GDN kernel indexes its q/k rows as
 * `iv1 % neq1`, and the unfused path gets the same mapping from ggml_repeat.
 * Blocked indexing is what this engine did while the 27B was being brought up,
 * and it left the mixer output at cosine 0.81 with every input exact.
 *
 *   gcc -I src tests/q35_heads_test.c -o build/q35_heads_test && build/q35_heads_test
 */

#include "nita.h"

#include <stdio.h>

static int failures;

static void expect(int h, int n_k_heads, int want)
{
    int got = nita_q35_k_head(h, n_k_heads);

    if (got != want) {
        printf("FAIL v-head %d over %d key heads -> %d, want %d\n",
               h, n_k_heads, got, want);
        ++failures;
    }
}

/* The reading this test exists to reject, kept so the two can be compared. */
static int blocked_head(int h, int n_v_heads, int n_k_heads)
{
    int grp = n_k_heads > 0 ? n_v_heads / n_k_heads : 0;

    return grp > 0 ? h / grp : h;
}

int main(void)
{
    int nv, nk, h, k, differ;

    /* Equal counts: both readings coincide, which is why the 2B passed. */
    for (h = 0; h < 16; ++h) {
        expect(h, 16, h);
        if (blocked_head(h, 16, 16) != nita_q35_k_head(h, 16)) {
            printf("FAIL equal counts: readings differ at v-head %d\n", h);
            ++failures;
        }
    }

    /* Bonsai 27B geometry: 48 value heads, 16 key heads. */
    nv = 48;
    nk = 16;
    for (h = 0; h < nv; ++h) expect(h, nk, h % nk);

    /* The heads where the two readings part company. */
    differ = 0;
    for (h = 0; h < nv; ++h) if (blocked_head(h, nv, nk) != nita_q35_k_head(h, nk)) ++differ;
    if (differ != 44) {
        printf("FAIL 48/16: expected 44 of 48 heads to distinguish the readings, got %d\n", differ);
        ++failures;
    }
    if (blocked_head(3, nv, nk) != 1 || nita_q35_k_head(3, nk) != 3) {
        printf("FAIL 48/16: v-head 3 should read key head 3, blocked reads %d\n",
               blocked_head(3, nv, nk));
        ++failures;
    }

    /* Small geometry, written out so the tiling is visible at a glance. */
    {
        static const int want4[4] = { 0, 1, 0, 1 };

        for (h = 0; h < 4; ++h) expect(h, 2, want4[h]);
    }

    /* Every key head has to feed the same number of value heads. */
    for (k = 0; k < nk; ++k) {
        int users = 0;

        for (h = 0; h < nv; ++h) if (nita_q35_k_head(h, nk) == k) ++users;
        if (users != nv / nk) {
            printf("FAIL key head %d feeds %d value heads, want %d\n", k, users, nv / nk);
            ++failures;
        }
    }

    /* A tiled map only depends on h modulo n_k_heads. */
    for (h = 0; h < 2 * nk; ++h) {
        if (nita_q35_k_head(h, nk) != nita_q35_k_head(h + nk, nk)) {
            printf("FAIL tiling: v-head %d and %d read different key heads\n", h, h + nk);
            ++failures;
        }
    }

    printf(failures ? "FAIL (%d failures)\n" : "PASS (0 failures)\n", failures);
    return failures ? 1 : 0;
}
