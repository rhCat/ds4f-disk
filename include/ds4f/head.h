/*
 * head.h -- output head + embedding + sampling (issue #6 step 3).
 *
 * head.json / embed.json are written by tools/convert-ds4f.py convert
 * from the checkpoint's head.weight/head.scale/embed.weight (bytes
 * copied as-is). Logits = head . state (F8_E4M3, E8M0 group scales).
 * Sampling is DETERMINISTIC: xorshift64 seeded by the caller, softmax
 * over the full vocab, single PRNG draw -- same seed, same token.
 */
#ifndef DS4F_HEAD_H
#define DS4F_HEAD_H

#include "ds4f/ds4f.h"

#include <stdint.h>

typedef struct Ds4fHead {
    uint8_t *buf;           /* head.bin mmap'd/read whole */
    long     buf_n;
    long     w_off, w_nbytes, s_off, s_nbytes;
    long     dims[4];
    int      rank, w_dtype;
    long     sdims[4];
    int      srank;
} Ds4fHead;

typedef struct Ds4fEmbed {
    uint8_t *buf;
    long     buf_n;
    long     dims[4];
    int      rank, dtype;
} Ds4fEmbed;

int ds4f_head_load(Ds4fHead *h, const char *json_path);
void ds4f_head_free(Ds4fHead *h);
int ds4f_embed_load(Ds4fEmbed *e, const char *json_path);
void ds4f_embed_free(Ds4fEmbed *e);

/* logits[V] = head . state[H]; V = head rows. */
int ds4f_head_logits(const Ds4fHead *h, const float *state, float *logits);

/* state[H] = embed row tok (F32 or BF16; F8 refused for now). */
int ds4f_embed_gather(const Ds4fEmbed *e, int tok, float *out);

/* Deterministic top-k-free softmax sample: mutates *rng. */
int ds4f_sample(const float *logits, int V, uint64_t *rng);

#endif /* DS4F_HEAD_H */
