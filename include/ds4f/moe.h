/*
 * moe.h -- trunk/pool tensor layouts + the real MoE compute step
 * (issue #2, milestone step 3).
 *
 * The layouts come from the converter (trunk.json) and the quantizer
 * (pool-mxfp4.json). The MoE step runs REAL math: router matvec on the
 * resident fp32 gate matrix, top-k selection, and the mxfp4 expert
 * chain w1 -> w2 -> w3 against the pool bytes the cache fetched.
 */
#ifndef DS4F_MOE_H
#define DS4F_MOE_H

#include "ds4f/ds4f.h"

#include <stdint.h>

#define DS4F_MAX_LAYERS 256
#define DS4F_MAX_TENSORS_PER_EXPERT 8

typedef struct Ds4fMoETensor {
    long dims[4];
    int  rank;
    long rel_v;         /* mxfp4 values offset, relative to slot start */
    long rel_s;         /* mxfp4 block scales offset, relative to slot */
    long v_nbytes, s_nbytes;
    int  bsize;         /* 16 or 32 */
} Ds4fMoETensor;

typedef struct Ds4fExpertLayout {
    int n;
    Ds4fMoETensor t[DS4F_MAX_TENSORS_PER_EXPERT];
} Ds4fExpertLayout;

typedef struct Ds4fPoolLayout {
    int      n_layers, n_experts;
    int64_t  expert_nbytes;
    int64_t  max_rc;            /* largest R*C over all expert tensors */
    Ds4fExpertLayout *exp;      /* [n_layers*n_experts] */
} Ds4fPoolLayout;

/* Load pool-mxfp4.json (as written by tools/convert-ds4f.py quantize). */
int ds4f_pool_layout_load(Ds4fPoolLayout *pl, const char *path,
                          const Ds4fCfg *cfg);

typedef struct Ds4fTrunkTensor {
    char name[96];
    int  dtype;                 /* 0=F32, 1=I8, 2=F8_E4M3, 4=BF16, 3=other */
    long dims[4];
    int  rank;
    long off;                   /* relative to layer payload start */
    long nbytes;
} Ds4fTrunkTensor;

typedef struct Ds4fTrunkLayout {
    int   n_layers;
    Ds4fTrunkTensor *t;         /* flat, layer-major */
    int  *t_off;                /* [n_layers+1] index into t */
    int   gate[DS4F_MAX_LAYERS]; /* tensor idx or -1 */
    int   gate_bias[DS4F_MAX_LAYERS];
    int   down[DS4F_MAX_LAYERS];
    int   up[DS4F_MAX_LAYERS];
    /* MLA attention roles (issue #6); _s = the E8M0 scale sibling */
    int   attn_qn[DS4F_MAX_LAYERS];
    int   attn_kvn[DS4F_MAX_LAYERS];
    int   attn_wqa[DS4F_MAX_LAYERS],  attn_wqa_s[DS4F_MAX_LAYERS];
    int   attn_wqb[DS4F_MAX_LAYERS],  attn_wqb_s[DS4F_MAX_LAYERS];
    int   attn_wkv[DS4F_MAX_LAYERS],  attn_wkv_s[DS4F_MAX_LAYERS];
    int   attn_woa[DS4F_MAX_LAYERS],  attn_woa_s[DS4F_MAX_LAYERS];
    int   attn_wob[DS4F_MAX_LAYERS],  attn_wob_s[DS4F_MAX_LAYERS];
    int   attn_woc[DS4F_MAX_LAYERS],  attn_woc_s[DS4F_MAX_LAYERS];
    int   attn_sink[DS4F_MAX_LAYERS];
    int   kvlat;                /* wkv output dim (0 = no attention) */
} Ds4fTrunkLayout;

/* Load trunk.json (as written by tools/convert-ds4f.py convert). */
int ds4f_trunk_layout_load(Ds4fTrunkLayout *tl, const char *path);

/* Top-k over scores: descending, tie-break by expert index (earlier
 * expert wins ties -- deterministic). */
void ds4f_topk(const float *scores, int E, int k, int *idx, float *w);

/* Real MoE step for layer L. state[hidden] in/out. tr = trunk layer
 * payload; es[j] = cache slot payload for sel[j] (topk, already
 * fetched). scratch holds max_rc floats; job_scratch[k] for k in
 * [0, topk) holds another max_rc floats each, allocated ONCE by the
 * caller (the parallel expert chains must not malloc per call --
 * fresh mmap'd scratch page-faults ~16 MB x topk x layers). Counters
 * accumulate. */
int ds4f_moe_step(const Ds4fCfg *cfg, const Ds4fTrunkLayout *tl, int L,
                  const uint8_t *tr, const Ds4fPoolLayout *pl,
                  const uint8_t *const *es, const int *sel, const float *wsel,
                  float *state, float *scratch, long scratch_n,
                  float *const *job_scratch,
                  int64_t *n_matvec, int64_t *n_decode);

#endif /* DS4F_MOE_H */
