#!/bin/bash
# Disassemble ds4f_moe_step around +428 to identify the detecting malloc.
set -u
cd ~/ds4f-disk
lldb -b -o "disassemble -n ds4f_moe_step" -o "quit" -- ./ds4f \
    2>&1 | awk '/ds4f_moe_step/,0' | head -60
