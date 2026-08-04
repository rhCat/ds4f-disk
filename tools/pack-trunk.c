/* pack-trunk.c -- rewrite N dense layer files into one contiguous
 * trunk.bin + a trunk.offsets table. Contiguity is what turns the
 * per-layer read into a single sequential read; a sparse file layout
 * would scatter the same bytes across seek boundaries. This is the
 * real-model path; make-fixture writes the same layout directly.
 *
 * usage: pack-trunk OUT_BIN OUT_OFFSETS layer_0.bin layer_1.bin ...
 */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
    if (argc < 4) {
        fprintf(stderr, "usage: %s OUT_BIN OUT_OFFSETS layer_0.bin ...\n", argv[0]);
        return 1;
    }
    const char *bin_path = argv[1];
    const char *off_path = argv[2];
    int n = argc - 3;
    if (n > 65536) { fprintf(stderr, "too many layers\n"); return 1; }

    /* size every input first */
    int64_t *sizes = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    if (!sizes) return 1;
    int64_t total = 0;
    for (int i = 0; i < n; i++) {
        int fd = open(argv[3 + i], O_RDONLY);
        if (fd < 0) { fprintf(stderr, "open %s: %s\n", argv[3 + i], strerror(errno)); return 1; }
        off_t sz = lseek(fd, 0, SEEK_END);
        close(fd);
        if (sz <= 0) { fprintf(stderr, "empty layer %s\n", argv[3 + i]); return 1; }
        sizes[i] = (int64_t)sz;
        total += sizes[i];
    }

    int ofd = open(off_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int bfd = open(bin_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0 || bfd < 0) { fprintf(stderr, "cannot create outputs\n"); return 1; }

    uint8_t hdr[8];
    put_u64(hdr, (uint64_t)n);
    if (write_all(ofd, hdr, 8)) return 1;

    uint8_t ob[16];
    uint8_t buf[1 << 20];
    int64_t at = 0;
    for (int i = 0; i < n; i++) {
        put_u64(ob, (uint64_t)at);
        put_u64(ob + 8, (uint64_t)sizes[i]);
        if (write_all(ofd, ob, 16)) return 1;
        int fd = open(argv[3 + i], O_RDONLY);
        if (fd < 0) return 1;
        int64_t left = sizes[i];
        while (left > 0) {
            ssize_t r = read(fd, buf, left > (int64_t)sizeof buf ? (int64_t)sizeof buf : left);
            if (r <= 0) { fprintf(stderr, "short read in %s\n", argv[3 + i]); return 1; }
            if (write_all(bfd, buf, (size_t)r)) return 1;
            left -= r;
        }
        close(fd);
        at += sizes[i];
    }
    close(ofd);
    close(bfd);

    printf("packed %d layers, %lld bytes -> %s + %s\n",
           n, (long long)total, bin_path, off_path);
    return 0;
}
