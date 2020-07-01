#!/bin/bash
source /etc/modules/openmpi              # Подключаем модуль с компилятором MPI.
source /etc/modules/gcc/9                # модуль с компилятором GCC
source /etc/modules/make                 # модуль с командой make
mpiCC t1.c -o t1
mpirun t1
