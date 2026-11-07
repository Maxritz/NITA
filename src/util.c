/*
 * util.c – growable buffers, allocation helpers and small path utilities.
 */

#include "nita.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Allocation – fail loudly rather than silently dereferencing NULL.   */
/* ------------------------------------------------------------------ */
static void oom(void)
{
    fprintf(stderr, "nita: out of memory\n");
    fflush(stderr);
    abort();
}

void *nita_malloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) oom();
    return p;
}

void *nita_calloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) oom();
    return p;
}

void *nita_realloc(void *p, size_t n)
{
    void *q = realloc(p, n ? n : 1);
    if (!q) oom();
    return q;
}

char *nita_strdup(const char *s)
{
    size_t n;
    char  *p;
    if (!s) return NULL;
    n = strlen(s);
    p = (char *)nita_malloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

/* ------------------------------------------------------------------ */
/* Dynamic buffer                                                      */
/* ------------------------------------------------------------------ */
void nbuf_init(nbuf_t *b)
{
    b->data = NULL;
    b->len  = 0;
    b->cap  = 0;
}

void nbuf_free(nbuf_t *b)
{
    if (!b) return;
    free(b->data);
    b->data = NULL;
    b->len  = 0;
    b->cap  = 0;
}

void nbuf_clear(nbuf_t *b)
{
    if (!b) return;
    b->len = 0;
    if (b->data && b->cap) b->data[0] = '\0';
}

static int nbuf_reserve(nbuf_t *b, size_t extra)
{
    size_t need = b->len + extra + 1; /* +1 for NUL */
    size_t cap;

    if (need <= b->cap) return 0;

    cap = b->cap ? b->cap : 128;
    while (cap < need) {
        if (cap > ((size_t)-1) / 2) { cap = need; break; }
        cap *= 2;
    }
    b->data = (char *)nita_realloc(b->data, cap);
    b->cap  = cap;
    return 0;
}

int nbuf_append(nbuf_t *b, const void *bytes, size_t n)
{
    if (!b) return -1;
    if (n == 0) return 0;
    if (nbuf_reserve(b, n) != 0) return -1;
    memcpy(b->data + b->len, bytes, n);
    b->len += n;
    b->data[b->len] = '\0';
    return 0;
}

int nbuf_puts(nbuf_t *b, const char *s)
{
    if (!s) return 0;
    return nbuf_append(b, s, strlen(s));
}

int nbuf_putc(nbuf_t *b, char c)
{
    return nbuf_append(b, &c, 1);
}

int nbuf_printf(nbuf_t *b, const char *fmt, ...)
{
    va_list ap, ap2;
    int     n;

    va_start(ap, fmt);
    va_copy(ap2, ap);
    n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) { va_end(ap2); return -1; }

    if (nbuf_reserve(b, (size_t)n) != 0) { va_end(ap2); return -1; }
    vsnprintf(b->data + b->len, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
    return 0;
}

char *nbuf_cstr(nbuf_t *b)
{
    if (!b) return NULL;
    if (!b->data) {
        nbuf_reserve(b, 0);
        b->data[0] = '\0';
    }
    return b->data;
}

/* ------------------------------------------------------------------ */
/* Paths and formatting                                                */
/* ------------------------------------------------------------------ */
const char *nita_basename(const char *path)
{
    const char *p, *base = path;
    if (!path) return "";
    for (p = path; *p; ++p) {
        if (*p == '/' || *p == '\\') base = p + 1;
    }
    return base;
}

const char *nita_human_bytes(uint64_t bytes)
{
    static char buf[64];
    static const char *unit[] = { "B", "KiB", "MiB", "GiB", "TiB" };
    double v = (double)bytes;
    int    u = 0;

    while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
    if (u == 0)
        snprintf(buf, sizeof buf, "%llu B", (unsigned long long)bytes);
    else
        snprintf(buf, sizeof buf, "%.2f %s", v, unit[u]);
    return buf;
}

uint64_t nita_fnv1a64(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint64_t       h = 0xcbf29ce484222325ULL;
    size_t         i;

    for (i = 0; i < n; ++i) {
        h ^= (uint64_t)b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}
