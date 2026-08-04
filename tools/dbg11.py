#!/usr/bin/env python3
# Reproduce test 11's F8 SIMD-vs-scalar mismatch in C with prints.
import subprocess, tempfile, os
src = r'''
#include "ds4f/kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(void) {
    uint8_t W[16*64]; uint8_t sc[16*2]; float x[64];
    srand(7);
    for (int i = 0; i < 16*64; i++) {
        uint8_t v = (uint8_t)(rand()%256);
        if (i % 5 == 0) v &= 0x87;
        if (i % 7 == 0) v |= 0x78;
        W[i] = v;
    }
    for (int i = 0; i < 16*2; i++) sc[i] = 117 + (uint8_t)(rand()%8);
    for (int i = 0; i < 64; i++) x[i] = (float)(rand()%200)/100.0f - 1.0f;
    float yv[16], ys[16];
    ds4f_kernels_set_simd(1);
    ds4f_f8_matvec(W, sc, 16, 64, 1, 1, x, yv);
    ds4f_kernels_set_simd(0);
    ds4f_f8_matvec(W, sc, 16, 64, 1, 1, x, ys);
    ds4f_kernels_set_simd(1);
    for (int r = 0; r < 16; r++) {
        double d = fabs((double)yv[r] - (double)ys[r]);
        double rel = d / (1.0 + fabs((double)ys[r]));
        if (rel > 1e-4) printf("r%2d vec=%.6g scl=%.6g rel=%g\n",
                               r, (double)yv[r], (double)ys[r], rel);
    }
    return 0;
}
'''
T = tempfile.mkdtemp()
open(f"{T}/t.c", "w").write(src)
subprocess.run(["cc", "-std=c99", "-O2", "-Iinclude", "-Isrc", "-o", f"{T}/t",
                f"{T}/t.c", "src/kernels.c", "src/simd.c", "-lm"],
               cwd=os.path.expanduser("~/ds4f-disk"), check=True)
subprocess.run([f"{T}/t"], cwd=os.path.expanduser("~/ds4f-disk"))
