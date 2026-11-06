/*
 * nita.h – shared declarations for the NITA engine.
 *
 * Modules:
 *   gguf.c      – GGUF v2/v3 container parser (metadata + tensor table)
 *   json.c      – minimal JSON parser / writer for the HTTP API
 *   tokenizer.c – vocabulary + greedy longest-match encoder/decoder
 *   chat.c      – conversation history + completion generation
 *   http.c      – cross-platform HTTP/1.1 server with OpenAI-style routes
 *   main.c      – CLI entry point
 */

#ifndef NITA_H
#define NITA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#define NITA_VERSION "0.1.0"

/* Opaque transformer handle, defined in transformer.c. */
typedef struct nita_model nita_model_t;

/* ------------------------------------------------------------------ */
/* Breadcrumbs – compile-time traced diagnostics (stderr)              */
/* ------------------------------------------------------------------ */
#ifdef DEBUG_BREADCRUMBS
#  define BREADCRUMB(tag, msg) \
        do { fprintf(stderr, "[breadcrumb] %s %s\n", (tag), (msg)); fflush(stderr); } while (0)
#  define BREADCRUMBF(tag, ...) \
        do { fprintf(stderr, "[breadcrumb] %s ", (tag)); \
             fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fflush(stderr); } while (0)
#else
#  define BREADCRUMB(tag, msg)      ((void)0)
#  define BREADCRUMBF(tag, ...)     ((void)0)
#endif

/* ------------------------------------------------------------------ */
/* Sherlock-lite trace – compact component table on stdout             */
/* ------------------------------------------------------------------ */
#ifdef SHERLOCK_LITE
#  define SHERLOCK_HEADER() \
        printf("%-14s%-8s%-10s%-12s%-12s%-10s\n", \
               "component", "ops", "%dev", "dev_us", "idle_us", "host_us")
#  define SHERLOCK_ROW(c, ops, dev, dev_us, idle_us, host_us) \
        printf("%-14s%-8d%-10.1f%-12.2f%-12.2f%-10.2f\n", \
               (c), (int)(ops), (double)(dev), (double)(dev_us), \
               (double)(idle_us), (double)(host_us))
#else
#  define SHERLOCK_HEADER() ((void)0)
#  define SHERLOCK_ROW(c, ops, dev, dev_us, idle_us, host_us) ((void)0)
#endif

/* ------------------------------------------------------------------ */
/* Growable byte buffer                                                */
/* ------------------------------------------------------------------ */
typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} nbuf_t;

void   nbuf_init(nbuf_t *b);
void   nbuf_free(nbuf_t *b);
void   nbuf_clear(nbuf_t *b);
int    nbuf_append(nbuf_t *b, const void *bytes, size_t n);
int    nbuf_puts(nbuf_t *b, const char *s);
int    nbuf_putc(nbuf_t *b, char c);
int    nbuf_printf(nbuf_t *b, const char *fmt, ...);
/* Ensure the buffer is NUL terminated and return the C string. */
char  *nbuf_cstr(nbuf_t *b);

/* ------------------------------------------------------------------ */
/* Utility                                                             */
/* ------------------------------------------------------------------ */
void  *nita_malloc(size_t n);
void  *nita_calloc(size_t n, size_t sz);
void  *nita_realloc(void *p, size_t n);
char  *nita_strdup(const char *s);
/* Path helpers */
const char *nita_basename(const char *path);
/* Format a byte count into a human readable string (static buffer). */
const char *nita_human_bytes(uint64_t bytes);

/* ================================================================== */
/* GGUF                                                                */
/* ================================================================== */

/* GGUF metadata value types (on-disk enumeration). */
enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
};

typedef struct {
    char    *key;
    uint32_t type;          /* GGUF_TYPE_* */
    /* Scalar payload (interpretation depends on type) */
    union {
        uint64_t u64;
        int64_t  i64;
        double   f64;
        struct { char *ptr; uint64_t len; } str;
    } v;
    /* Array payload */
    uint32_t item_type;     /* GGUF_TYPE_* of each element */
    uint64_t count;
    uint8_t *raw;           /* serialized elements */
} gguf_kv_t;

typedef struct {
    char     *name;
    uint32_t  n_dims;
    uint64_t  dims[4];
    uint32_t  type;         /* ggml_type */
    uint64_t  offset;       /* offset from start of data section */
    uint64_t  n_elements;   /* product of dims */
} gguf_tensor_t;

typedef struct {
    char          *path;
    uint32_t       version;
    uint64_t       tensor_count;
    uint64_t       kv_count;
    gguf_kv_t     *kv;
    gguf_tensor_t *tensors;
    uint64_t       data_offset;   /* absolute file offset of tensor data */
    uint64_t       file_size;
    uint8_t       *data;          /* whole file image (owned) */

    /* Sequential cursor for gguf_array_str(). Vocabularies hold hundreds of
     * thousands of entries and are always read in order, so walking from the
     * start on every call would be quadratic. */
    const gguf_kv_t *arr_cursor_kv;
    uint64_t         arr_cursor_i;
    uint64_t         arr_cursor_pos;
} gguf_file_t;

/* Load a whole .gguf file. Returns NULL and fills err on failure. */
gguf_file_t *gguf_load(const char *path, char *err, size_t errlen);
void         gguf_free(gguf_file_t *g);

/* Metadata accessors (NULL / defaults when absent). */
const char *gguf_get_str(const gguf_file_t *g, const char *key, const char *fallback);
int64_t     gguf_get_int(const gguf_file_t *g, const char *key, int64_t fallback);
double      gguf_get_float(const gguf_file_t *g, const char *key, double fallback);
bool        gguf_has_key(const gguf_file_t *g, const char *key);

/* Array accessors. For string arrays, use gguf_array_str().
 * Returns element count, or 0 when the key is missing / wrong type. */
uint64_t    gguf_array_count(const gguf_file_t *g, const char *key);
/* Fetch element index of a string array. Returns NULL if out of range. */
const char *gguf_array_str(const gguf_file_t *g, const char *key, uint64_t index);
/* Fetch element index of a numeric array as double. */
double      gguf_array_num(const gguf_file_t *g, const char *key, uint64_t index);

/* Human readable name for a ggml tensor type id. */
const char *ggml_type_name(uint32_t type);
/* Bytes occupied by one row (one element-block group) for a ggml type. */
uint32_t    ggml_type_size(uint32_t type);
uint32_t    ggml_blck_size(uint32_t type);

/* ================================================================== */
/* JSON                                                                */
/* ================================================================== */

typedef enum {
    JSON_NULL = 0,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} json_type_t;

typedef struct json_value json_value_t;

struct json_value {
    json_type_t type;
    union {
        bool     b;
        double   num;
        struct { char *ptr; size_t len; } str;
        struct { json_value_t **items; size_t count; } arr;
        struct { char **keys; json_value_t **vals; size_t count; } obj;
    } u;
};

/* Parse a JSON document. Returns NULL on error and fills err. */
json_value_t *json_parse(const char *text, size_t len, char *err, size_t errlen);
void          json_free(json_value_t *v);

json_value_t *json_object_get(const json_value_t *obj, const char *key);
const char   *json_string(const json_value_t *v);        /* NULL unless JSON_STRING */
double        json_number(const json_value_t *v, double fallback);
bool          json_bool(const json_value_t *v, bool fallback);
size_t        json_array_count(const json_value_t *v);
json_value_t *json_array_at(const json_value_t *v, size_t i);

/* Append a JSON-escaped string (with surrounding quotes) to buf. */
void json_escape(nbuf_t *out, const char *s, size_t len);

/* ================================================================== */
/* Tokenizer                                                           */
/* ================================================================== */

typedef enum {
    TOK_MODE_GREEDY = 0,  /* longest-match fallback            */
    TOK_MODE_BPE    = 1,  /* GPT-2 style byte-level BPE+merges */
    TOK_MODE_SPM    = 2   /* llama/sentencepiece score merges  */
} tok_mode_t;

typedef struct {
    char     **tokens;      /* token id -> token string */
    float     *scores;      /* token id -> score (optional) */
    uint32_t   n_tokens;
    int32_t    id_bos, id_eos, id_unk;
    /* open-addressing hash table: token string -> id */
    int32_t   *slots;       /* id, or -1 */
    uint32_t   n_slots;
    bool       has_vocab;

    tok_mode_t   mode;
    const char  *mode_name;

    /* merge ranks keyed by "left right" -> rank (lower wins) */
    char     **merge_keys;
    uint32_t  *merge_rank;
    uint32_t   merge_cap;
    uint32_t   n_merges;

    /* Raw byte -> single token id, used when a byte has no vocabulary entry. */
    int32_t    byte_token_id[256];
} tokenizer_t;

/* Build a tokenizer from GGUF metadata. Always returns a valid object
 * (empty/has_vocab=false when the model carries no vocabulary). */
tokenizer_t *tokenizer_from_gguf(const gguf_file_t *g);
void         tokenizer_free(tokenizer_t *t);

/* Encode text to token ids. Returns malloc'd array (caller frees). */
int32_t     *tokenizer_encode(const tokenizer_t *t, const char *text,
                              size_t *out_count);
/* Decode token ids back to a malloc'd NUL-terminated string. */
char        *tokenizer_decode(const tokenizer_t *t, const int32_t *ids, size_t count);
/* Append the printable form of one token to a text stream. */
void         tokenizer_append_piece(const tokenizer_t *t, int32_t id, nbuf_t *out);

/* ================================================================== */
/* Chat                                                                */
/* ================================================================== */

#define CHAT_MAX_MESSAGES 128

typedef enum { ROLE_SYSTEM = 0, ROLE_USER, ROLE_ASSISTANT } chat_role_t;

typedef struct {
    chat_role_t role;
    char       *content;
} chat_message_t;

typedef struct {
    chat_message_t messages[CHAT_MAX_MESSAGES];
    size_t         count;
} chat_history_t;

void chat_history_init(chat_history_t *h);
void chat_history_free(chat_history_t *h);
void chat_history_clear(chat_history_t *h);
int  chat_history_push(chat_history_t *h, chat_role_t role, const char *content);
/* Render the conversation into a prompt string using the model's template. */
char *chat_render_prompt(const gguf_file_t *g, const chat_history_t *h);

/* Result of a completion request. */
typedef struct {
    char       *content;      /* malloc'd assistant text */
    size_t      prompt_tokens;
    size_t      completion_tokens;
    const char *finish_reason;/* static string: "stop" or "length" */
    const char *fingerprint;  /* "fp_nita" for real generation, "fp_nita_stub" otherwise */
} chat_completion_t;

/* Run a completion over the history using the transformer engine.
 * Falls back to an acknowledgement when no usable weights are loaded. */
int chat_complete(const gguf_file_t *g, const tokenizer_t *tok,
                  nita_model_t *engine,
                  const chat_history_t *h, int max_tokens,
                  float temperature, int top_k,
                  chat_completion_t *out);
void chat_completion_free(chat_completion_t *c);

/* ================================================================== */
/* Dequantisation                                                      */
/* ================================================================== */

float  nita_f16_to_f32(uint16_t h);
/* Bytes consumed by one row of ne0 elements. */
size_t nita_row_bytes(uint32_t type, uint64_t ne0);
/* Expand one row of packed GGUF data into floats. 0 on success, -1 if the
 * tensor type is not supported. */
int    nita_dequant_row(const uint8_t *src, uint32_t type, uint64_t ne0, float *dst);

/* ================================================================== */
/* Transformer                                                         */
/* ================================================================== */

typedef struct {
    const uint8_t *data;
    uint32_t       type;
    uint64_t       ne0;      /* input width  */
    uint64_t       ne1;      /* output height */
    bool           bound;
} nita_tensor_t;

/* Build the engine from a parsed GGUF file. Returns NULL and fills err when
 * the file is not a supported llama-style transformer. */
nita_model_t *nita_model_open(const gguf_file_t *g, int ctx, char *err, size_t errlen);
void          nita_model_close(nita_model_t *m);
/* Drop the KV cache so a new conversation starts at position 0. */
void          nita_model_reset(nita_model_t *m);
int           nita_model_ctx(const nita_model_t *m);
int           nita_model_n_vocab(const nita_model_t *m);

/* Evaluate one token, appending it to the KV cache. logits must hold
 * n_vocab floats. Returns 0 on success. */
int nita_forward(nita_model_t *m, int32_t token, float *logits);

/* Sample an id from logits using temperature and optional top-k. */
int32_t nita_sample(const float *logits, int n_vocab,
                    float temperature, int top_k, uint32_t *rng);

/* Prefill the prompt then sample up to max_tokens new ids, stopping at eos.
 * Returns the number of ids produced (0 on failure). */
int nita_generate(nita_model_t *m, const int32_t *prompt, size_t n_prompt,
                  int max_tokens, float temperature, int top_k, int32_t eos_id,
                  int32_t **out_ids, size_t *out_count, uint32_t seed,
                  bool *hit_limit);

/* ================================================================== */
/* HTTP server                                                         */
/* ================================================================== */

typedef struct {
    const char     *model_path;
    gguf_file_t    *model;
    tokenizer_t    *tok;
    nita_model_t   *engine;      /* transformer; NULL when unavailable */
    const char     *model_id;    /* served model name */
    const char     *quant_name;  /* selected quantisation type */
    int             ctx_size;
} nita_server_t;

/* Start a blocking HTTP server. Returns 0 on clean shutdown. */
int http_serve(const char *host, int port, nita_server_t *srv);

#endif /* NITA_H */
