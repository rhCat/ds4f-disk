/* gate: trunk pin+ring streaming. Odd layer sizes exercise the table;
 * pinned bytes must match and streamed bytes must arrive intact. */
#include "ds4f/ds4f.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void put_u64(uint8_t *b, uint64_t v) {
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i));
}

static uint64_t fill_checksum(int64_t n, int layer) {
    uint64_t h = UINT64_C(0xCBF29CE484222325);
    for (int64_t i = 0; i < n; i++) {
        uint8_t v = (uint8_t)((i * 31 + layer * 17) & 0xFF);
        h = (h ^ v) * UINT64_C(0x100000001B3);
    }
    return h;
}

int main(void) {
    const int nlayers = 3;
    const int64_t sizes[3] = {1000, 2001, 3002};   /* deliberately odd */
    char dir[] = "/tmp/ds4f_trunk_XXXXXX";
    if (!mkdtemp(dir)) return 1;

    char bpath[512], opath[512];
    snprintf(bpath, sizeof bpath, "%s/trunk.bin", dir);
    snprintf(opath, sizeof opath, "%s/trunk.offsets", dir);

    int bfd = open(bpath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (bfd < 0) return 1;
    uint8_t *buf = (uint8_t *)malloc(3002);
    for (int L = 0; L < nlayers; L++) {
        for (int64_t i = 0; i < sizes[L]; i++)
            buf[i] = (uint8_t)((i * 31 + L * 17) & 0xFF);
        if (write(bfd, buf, (size_t)sizes[L]) != (ssize_t)sizes[L]) return 1;
    }
    close(bfd);

    int ofd = open(opath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0) return 1;
    uint8_t ob[16];
    put_u64(ob, (uint64_t)nlayers);
    if (write(ofd, ob, 8) != 8) return 1;
    int64_t at = 0;
    for (int L = 0; L < nlayers; L++) {
        put_u64(ob, (uint64_t)at);
        put_u64(ob + 8, (uint64_t)sizes[L]);
        if (write(ofd, ob, 16) != 16) return 1;
        at += sizes[L];
    }
    close(ofd);

    Ds4fTrunk tr;
    if (ds4f_trunk_open(&tr, bpath, opath) != 0) {
        fprintf(stderr, "open failed\n");
        return 1;
    }
    int npin;
    int64_t slot;
    /* budget must cover the ring first (2 x 3002), then pin layer 0 */
    ds4f_trunk_plan(&tr, 9000, &npin, &slot, 2);
    if (npin != 1) {
        fprintf(stderr, "plan pinned %d, want 1\n", npin);
        return 1;
    }
    if (ds4f_trunk_start(&tr, npin, slot, 2) != 0) {
        fprintf(stderr, "start failed\n");
        return 1;
    }
    for (int L = 0; L < nlayers; L++) {
        const uint8_t *p = ds4f_trunk_bind(&tr, L);
        if (!p) {
            fprintf(stderr, "bind %d failed\n", L);
            return 1;
        }
        if (ds4f_checksum(p, sizes[L]) != fill_checksum(sizes[L], L)) {
            fprintf(stderr, "layer %d content mismatch\n", L);
            return 1;
        }
    }
    if (tr.nread != sizes[1] + sizes[2]) {
        fprintf(stderr, "nread %lld, want %lld\n",
                (long long)tr.nread, (long long)(sizes[1] + sizes[2]));
        return 1;
    }
    ds4f_trunk_close(&tr);
    free(buf);
    unlink(bpath);
    unlink(opath);
    rmdir(dir);
    return 0;
}
