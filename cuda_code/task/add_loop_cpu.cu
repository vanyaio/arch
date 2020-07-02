/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "stdio.h"

#define N   10

__global__
void add( int *a, int *b, int *c ) {
    int tid = 0;    // this is CPU zero, so we start at zero
    while (tid < N) {
		/*
		 * printf("i = %d, a = %d, b = %d\n", tid, a[tid], b[tid]);
		 */
        c[tid] = a[tid] + b[tid];
        tid += 1;   // we have one CPU, so we increment by one
    }
}

int main( void ) {
    int a[N], b[N], c[N];

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

	int *ad, *bd, *cd;
	cudaMalloc((void**)ad, sizeof(int) * N);
	cudaMalloc((void**)bd, sizeof(int) * N);
	cudaMalloc((void**)cd, sizeof(int) * N);
	cudaMemcpy(ad, a, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(cd, c, sizeof(int) * N, cudaMemcpyHostToDevice);

    add<<<1, 1>>>(ad, bd, cd);
	cudaDeviceSynchronize();

	/*
     * for (int i=0; i<N; i++) {
     *     a[i] = 0;
     *     b[i] = 0;
     * }
	 */
	cudaMemcpy(a, ad, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(b, bd, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, cd, sizeof(int) * N, cudaMemcpyDeviceToHost);
    // display the results
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

    return 0;
}
