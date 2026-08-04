/* make-fixture.c -- build a tiny self-contained model directory so the
 * engine can be exercised WITHOUT any real weights: config.json, a real
 * safetensors file (header index + expert payloads), and a packed trunk
 * with its offsets table. Payload bytes are deterministic splitmix64. */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t mix64(uint64_t x) {
    x += UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static int write_all(int fd, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return -1;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

static void put_u64(uint8_t *b, uint64_t v) {
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i));
}

int main(int argc, char **argv) {
    const char *dir = "fixture";
    int n_layers = 6, n_experts = 24, topk = 3, hidden = 128, latent = 64;
    int moe_inter = 128, n_shared = 1;
    int64_t expert_bytes = 8192, trunk_bytes = 16384;
    uint64_t seed = 7;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) n_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) n_experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc) hidden = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--latent") && i + 1 < argc) latent = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--moe-inter") && i + 1 < argc) moe_inter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--expert-bytes") && i + 1 < argc) expert_bytes = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--trunk-bytes") && i + 1 < argc) trunk_bytes = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (expert_bytes <= 0 || trunk_bytes <= 0 || n_layers <= 0 || n_experts <= 0) {
        fprintf(stderr, "nonsense sizes\n");
        return 1;
    }

    char path[4096];
    snprintf(path, sizeof path, "%s", dir);
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "mkdir %s: %s\n", path, strerror(errno));
        return 1;
    }

    /* config.json */
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 1; }
    fprintf(f,
        "{\n"
        "  \"n_layers\": %d,\n"
        "  \"n_experts\": %d,\n"
        "  \"topk\": %d,\n"
        "  \"n_shared\": %d,\n"
        "  \"hidden\": %d,\n"
        "  \"latent\": %d,\n"
        "  \"moe_inter\": %d,\n"
        "  \"expert_nbytes\": %lld,\n"
        "  \"seed\": %llu\n"
        "}\n",
        n_layers, n_experts, topk, n_shared, hidden, latent, moe_inter,
        (long long)expert_bytes, (unsigned long long)seed);
    fclose(f);

    /* model.safetensors: 8-byte LE header length, JSON header, payload.
     * Expert "e.{layer}.{expert}" lives at layer-major expert-minor
     * offset, each expert_bytes long -- fixed-rate, O(1) addressable. */
    size_t cap = 4096;
    char *hdr = (char *)malloc(cap);
    if (!hdr) return 1;
    size_t hl = 0;
#define APPEND(...) do {                                                \
        int need = snprintf(NULL, 0, __VA_ARGS__);                      \
        if (need < 0 || hl + (size_t)need + 1 > cap) {                  \
            while (hl + (size_t)need + 1 > cap) cap *= 2;               \
            char *nh = (char *)realloc(hdr, cap);                       \
            if (!nh) return 1;                                          \
            hdr = nh;                                                   \
        }                                                               \
        hl += (size_t)snprintf(hdr + hl, cap - hl, __VA_ARGS__);        \
    } while (0)
    APPEND("{");
    for (int L = 0; L < n_layers; L++) {
        for (int e = 0; e < n_experts; e++) {
            long long a = (long long)L * n_experts * expert_bytes +
                          (long long)e * expert_bytes;
            APPEND("%s\"e.%d.%d\":{\"dtype\":\"U8\",\"shape\":[%lld],"
                   "\"data_offsets\":[%lld,%lld]}",
                   (L == 0 && e == 0) ? "" : ",",
                   L, e, (long long)expert_bytes, a, a + (long long)expert_bytes);
        }
    }
    APPEND("}");
#undef APPEND

    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 1; }
    uint8_t lenb[8];
    put_u64(lenb, (uint64_t)hl);
    if (write_all(fd, lenb, 8) || write_all(fd, hdr, hl)) return 1;
    uint8_t tmp[8192];
    uint64_t x = seed;
    int64_t total = (int64_t)n_layers * n_experts * expert_bytes;
    int64_t left = total;
    while (left > 0) {
        int64_t chunk = left > (int64_t)sizeof tmp ? (int64_t)sizeof tmp : left;
        for (int64_t i = 0; i < chunk; i += 8) {
            x = mix64(x);
            for (int k = 0; k < 8 && i + k < chunk; k++)
                tmp[i + k] = (uint8_t)(x >> (8 * k));
        }
        if (write_all(fd, tmp, (size_t)chunk)) return 1;
        left -= chunk;
    }
    close(fd);

    /* trunk.bin + trunk.offsets: equal-size layers for the fixture.
     * pack-trunk.c produces the same layout from real layer files. */
    snprintf(path, sizeof path, "%s/trunk.bin", dir);
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;
    left = (int64_t)n_layers * trunk_bytes;
    while (left > 0) {
        int64_t chunk = left > (int64_t)sizeof tmp ? (int64_t)sizeof tmp : left;
        for (int64_t i = 0; i < chunk; i += 8) {
            x = mix64(x ^ UINT64_C(0x7472756E6B));
            for (int k = 0; k < 8 && i + k < chunk; k++)
                tmp[i + k] = (uint8_t)(x >> (8 * k));
        }
        if (write_all(fd, tmp, (size_t)chunk)) return 1;
        left -= chunk;
    }
    close(fd);

    snprintf(path, sizeof path, "%s/trunk.offsets", dir);
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;
    uint8_t ob[16];
    put_u64(ob, (uint64_t)n_layers);
    if (write_all(fd, ob, 8)) return 1;
    for (int L = 0; L < n_layers; L++) {
        put_u64(ob, (uint64_t)((int64_t)L * trunk_bytes));
        put_u64(ob + 8, (uint64_t)trunk_bytes);
        if (write_all(fd, ob, 16)) return 1;
    }
    close(fd);

    printf("fixture at %s: %d layers x %d experts (topk %d), expert %lld B, "
           "trunk layer %lld B, seed %llu\n",
           dir, n_layers, n_experts, topk, (long long)expert_bytes,
           (long long)trunk_bytes, (unsigned long long)seed);
    return 0;
}
