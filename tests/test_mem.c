/* gate: memory planner refuses over budget, accepts under budget,
 * and the plan is a forecast (peak RSS is measured afterwards). */
#include "ds4f/ds4f.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    if (setenv("DS4F_TEST_MEM", "1073741824", 1) != 0) return 1;   /* 1 GB */

    Ds4fCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.hidden = 128;
    cfg.n_shared = 1;
    cfg.n_layers = 6;
    cfg.expert_nbytes = 8192;

    Ds4fMemPlan p;
    /* a 100 GB expert cache cannot fit in 1 GB -> refuse */
    ds4f_mem_plan(&p, &cfg, 0, (int64_t)100e9, 0, 1, 16384, 2);
    if (!ds4f_mem_refuses(&p)) {
        fprintf(stderr, "plan should refuse 100 GB on a 1 GB machine\n");
        return 1;
    }
    /* a 64 MB cache fits -> accept */
    ds4f_mem_plan(&p, &cfg, 0, (int64_t)64 << 20, 0, 0, 4096, 2);
    if (ds4f_mem_refuses(&p)) {
        fprintf(stderr, "plan should accept a small cache\n");
        return 1;
    }
    /* the measurement exists and is nonzero */
    if (ds4f_peak_rss() <= 0) {
        fprintf(stderr, "peak RSS measurement broken\n");
        return 1;
    }
    return 0;
}
