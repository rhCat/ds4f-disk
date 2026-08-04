/* mem.c -- the plan is a forecast, not a result (invariant 5). */
#include "ds4f/ds4f.h"

#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <stdio.h>
#endif

int64_t ds4f_mem_available(void) {
    const char *test = getenv("DS4F_TEST_MEM");
    if (test) return (int64_t)atoll(test);

#if defined(__APPLE__)
    uint64_t m = 0;
    size_t n = sizeof m;
    if (sysctlbyname("hw.memsize", &m, &n, NULL, 0) == 0)
        return (int64_t)m;                   /* physical; macOS has no MemAvailable */
    return (int64_t)8 << 30;
#else
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return (int64_t)8 << 30;
    char line[256];
    unsigned long long kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    fclose(f);
    return kb ? (int64_t)kb * 1024 : (int64_t)8 << 30;
#endif
}

void ds4f_mem_plan(Ds4fMemPlan *p, const Ds4fCfg *cfg, int64_t trunk_budget,
                   int64_t cache_bytes, int64_t shared_bytes, int npin,
                   int64_t slot, int nring) {
    (void)npin;   /* real pinned bytes are patched in by the caller */
    memset(p, 0, sizeof *p);
    p->trunk_pin_b = (double)(trunk_budget > 0 ? trunk_budget : 0);
    p->trunk_ring_b = (double)slot * (double)nring;
    p->cache_b     = (double)cache_bytes;
    p->shared_b    = (double)shared_bytes;
    p->state_b     = (double)cfg->hidden * 8.0 + 1e6;   /* activations + scratch */
    p->index_b     = 1e6;                                /* pointer map, small */
    p->need_b = p->trunk_pin_b + p->trunk_ring_b + p->cache_b +
                p->shared_b + p->state_b + p->index_b;
    p->have_b = (double)ds4f_mem_available();
}

int ds4f_mem_refuses(const Ds4fMemPlan *p) {
    return p->need_b > p->have_b * 0.95;
}

void ds4f_mem_print(const Ds4fMemPlan *p) {
    fprintf(stderr,
            "MEMORY PLAN (forecast, not result)\n"
            "  trunk pinned prefix   %9.1f GB\n"
            "  trunk ring            %9.1f GB\n"
            "  expert cache          %9.1f GB\n"
            "  shared experts (res)  %9.1f GB\n"
            "  state + scratch       %9.1f GB\n"
            "  pointer map           %9.1f GB\n"
            "  TOTAL need            %9.1f GB  vs  available %9.1f GB\n",
            p->trunk_pin_b / 1e9, p->trunk_ring_b / 1e9, p->cache_b / 1e9,
            p->shared_b / 1e9, p->state_b / 1e9, p->index_b / 1e9,
            p->need_b / 1e9, p->have_b / 1e9);
}

int64_t ds4f_peak_rss(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__)
    return (int64_t)ru.ru_maxrss;            /* bytes on macOS */
#else
    return (int64_t)ru.ru_maxrss * 1024;     /* KB on Linux */
#endif
}

uint64_t ds4f_checksum(const uint8_t *p, int64_t n) {
    uint64_t h = UINT64_C(0xCBF29CE484222325);
    for (int64_t i = 0; i < n; i++) h = (h ^ p[i]) * UINT64_C(0x100000001B3);
    return h;
}
