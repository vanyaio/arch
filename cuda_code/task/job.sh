#!/bin/bash
source /etc/modules/cuda/10.1
source /etc/modules/gcc/8
source /etc/modules/make
nvcc add_loop_cpu.cu -o test
./test
