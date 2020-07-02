#include "stdio.h"

#define N 4
#define M 4
#define thx 2
#define thy 2

__global__ void add( int *a, int *b, int *c )
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int ind = i * N + j;
	c[ind] = a[ind] + b[ind];
}

int main() {
	int *a, *b, *c;
	cudaMallocManaged(&a, M*N*sizeof(int));
	cudaMallocManaged(&b, M*N*sizeof(int));
	cudaMallocManaged(&c, M*N*sizeof(int));

	for (int i = 0; i < M * N; i++)
	{
		a[i] = i;
		b[i] = 2 * i;
	}

	dim3 blocks(N / thx, M / thy);
	dim3 threads(thx, thy);

	add<<< blocks, threads >>>(a, b, c);
	cudaDeviceSynchronize();

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++)
			printf("%d ", c[i*N + j]);
		printf("\n");
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}
