/*
 * attn.h -- MLA attention step + KV cache (issue #6, step 2).
 *
 * The real V4-Flash attention is MLA (multi-head latent attention):
 *   ql = wq_a . x            (q latent)
 *   qn = RMSNorm(ql, q_norm)
 *   q  = wq_b . qn           (up-project to full width)
 *   kv = RMSNorm(wkv . x, kv_norm)      (kv latent, cached per token)
 *   k  = kv[:kvlat/2], v = kv[kvlat/2:]
 *   scores = q . k / sqrt(d) + sink[pos] (sink positions always attend)
 *   out    = softmax(scores) . v
 *   state += wo_c . wo_b . wo_a . out
 *
 * The KV cache is per-layer (each layer computes its own kv latent);
 * MLA's shared-cache memory optimization is the roadmap refinement.
 * All tensors come from the trunk layout (F8 + E8M0 group scales,
 * BF16 norms, F32 sink). Attention is skipped for layers whose graph
 * is incomplete, so the engine degrades gracefully.
 */
#ifndef DS4F_ATTN_H
#define DS4F_ATTN_H

#include "ds4f/ds4f.h"
#include "ds4f/moe.h"

typedef struct Ds4fKvCache {
    float *kv;              /* [n_layers * max_tokens * kvlat] */
    int    n_layers, kvlat, max_tokens;
} Ds4fKvCache;

int  ds4f_kv_init(Ds4fKvCache *c, int n_layers, int kvlat, int max_tokens);
void ds4f_kv_free(Ds4fKvCache *c);

/* Attention for layer L. tr = trunk layer payload; state[hidden] in/out
 * (residual added). token = current token index (cache position).
 * Returns 0 (ok, possibly skipped), -1 on layout/config error. */
int ds4f_attn_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                   const uint8_t *tr, float *state, Ds4fKvCache *kv,
                   int token);

#endif /* DS4F_ATTN_H */
