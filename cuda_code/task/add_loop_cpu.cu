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
void add11( int *a, int *b, int *c ) {
    int i = 0;    // this is CPU zero, so we start at zero
    while (i < N) {
        c[i] = a[i] + b[i];
        i += 1;   // we have one CPU, so we increment by one
    }
}

__global__
void addn1( int *a, int *b, int *c ) {
	int i = blockIdx.x;
	c[i] = a[i] + b[i];
}

__global__
void add1n( int *a, int *b, int *c ) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__
void add( int *a, int *b, int *c ) {
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < N; i += stride)
		c[i] = a[i] + b[i];
}

int main( void ) {
	int *a, *b, *c;

	// fill the arrays 'a' and 'b' on the CPU
	cudaMallocManaged(&a, N*sizeof(int));
	cudaMallocManaged(&b, N*sizeof(int));
	cudaMallocManaged(&c, N*sizeof(int));
	for (int i=0; i<N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	add11<<<1, 1>>>(a, b, c);
	cudaDeviceSynchronize();

	add1n<<<1, 1>>>(a, b, c);
	cudaDeviceSynchronize();

	addn1<<<1, 1>>>(a, b, c);
	cudaDeviceSynchronize();

	addn1<<<1, 1>>>(a, b, c);
	cudaDeviceSynchronize();

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	add<<<numBlocks, blockSize>>>(a, b, c);
	cudaDeviceSynchronize();

	// display the results
	for (int i=0; i<N; i++) {
		printf( "%d + %d = %d\n", a[i], b[i], c[i] );
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	return 0;
}
