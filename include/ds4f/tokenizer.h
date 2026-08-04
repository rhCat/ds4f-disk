/*
 * tokenizer.h -- HF tokenizer.json support for ds4f-disk (issue #6
 * step 5): byte-level BPE decode (ids -> text) and encode (text ->
 * ids). Handles the gpt-2/DeepSeek byte<->unicode mapping, the vocab
 * object, the merges list, and plain-string vocabs (non-byte-level).
 *
 * Zero deps beyond libc; the JSON is parsed by the vendored
 * src/json.h (included by tokenizer.c).
 */
#ifndef DS4F_TOKENIZER_H
#define DS4F_TOKENIZER_H

#include <stdint.h>

typedef struct Ds4fTokenizer {
    int      nvocab;              /* vocab size */
    char   **vocab;               /* id -> token string (unicode form) */
    /* string -> id hash */
    char   **vkeys;               /* table: NUL-terminated token */
    int     *vids;
    int      vcap;                /* power of two */
    /* pair (left_id, right_id) -> (rank, merged_id) */
    uint64_t *pkeys;              /* (left<<32)|right */
    int32_t *pranks, *pmerged;
    int      pcap;
    /* byte<->unicode tables (gpt-2 style) */
    uint8_t rev[0x180];           /* char -> byte (0xFF = plain char) */
    uint16_t fwd[256];            /* byte -> char codepoint (>= 0x100 for
                                     the unsafe bytes) */
    int      nbytes_unsafe;       /* how many bytes map to 0x100+ */
} Ds4fTokenizer;

/* Load a tokenizer.json. Returns 0 on success. */
int ds4f_tokenizer_load(Ds4fTokenizer *t, const char *path);
void ds4f_tokenizer_free(Ds4fTokenizer *t);

/* Encode UTF-8 text to token ids (byte-level BPE). Returns the number
 * of ids (<= max_ids) or -1 on failure. */
int ds4f_tokenizer_encode(const Ds4fTokenizer *t, const char *utf8,
                          int *ids, int max_ids);

/* Decode token ids to UTF-8 text. Returns bytes written (<= out_n-1,
 * always NUL-terminated) or -1 on failure. */
int ds4f_tokenizer_decode(const Ds4fTokenizer *t, const int *ids, int n,
                          char *out, int out_n);

#endif /* DS4F_TOKENIZER_H */
