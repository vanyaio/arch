#include "stdio.h"
typedef struct {
	int N;
	int M;
	int step; 
	float* m;
} mat;


#define get_elem(A, r, c) (A.m[r * A.step + c])
/*
 * __device__ float GetElement(const mat A, int row, int col)
 * {
 *     return A.m[row * A.step + col];
 * }
 */

#define set_elem(A, r, c, val) A.m[r * A.step + c] = val
/*
 * __device__ void SetElement(mat A, int row, int col, float value)
 * {
 *     A.m[row * A.step + col] = value;
 * }
 */

#define BLOCK_SIZE 2

__device__ mat get_submat(mat A, int row, int col) 
{
	mat Asub;
	Asub.M = BLOCK_SIZE;
	Asub.N = BLOCK_SIZE;
	Asub.step = A.step;
	Asub.m = &A.m[A.step * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}


__global__ void matmul(mat A, mat B, mat C)
{
	int blockCol = blockIdx.x;
	int blockRow = blockIdx.y;

	mat Csub = get_submat(C, blockRow, blockCol);

	int row = threadIdx.y;
	int col = threadIdx.x;
	float cij = 0;
	for (int m = 0; m < (A.M / BLOCK_SIZE); ++m) {
		mat Asub = get_submat(A, blockRow, m);

		mat Bsub = get_submat(B, m, blockCol);

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[row][col] = get_elem(Asub, row, col);
		Bs[row][col] = get_elem(Bsub, row, col);

		__syncthreads();
		for (int e = 0; e < BLOCK_SIZE; ++e)
		cij += As[row][e] * Bs[e][col];

		__syncthreads();
	}

	set_elem(Csub, row, col, cij);
}

int main()
{
	float m1[] = {1,2,3,4,
				5,6,7,8,
				9,10,11,12,
				13,14,15,16};

	float m2[] = {16,15,14,13,
				12,11,10,9,
				8,7,6,5,
				4,3,2,1};

	float *m3 = (float*)malloc(4 * 4 * sizeof(float));

	mat A = { .N = 4, .M = 4, .step = 4, .m = m1 };
	mat B = { .N = 4, .M = 4, .step = 4, .m = m2 };
	mat C = { .N = 4, .M = 4, .step = 4, .m = m3 };

	mat d_A = { .N = A.N, .M = A.M, .step = A.M };
	int size = A.M * A.N * sizeof(float);
	cudaMalloc(&d_A.m, size);
	cudaMemcpy(d_A.m, A.m, size, cudaMemcpyHostToDevice);

	mat d_B = { .N = B.N, .M = B.M, .step = B.M };
	size = B.M * B.N * sizeof(float);
	cudaMalloc(&d_B.m, size);
	cudaMemcpy(d_B.m, B.m, size, cudaMemcpyHostToDevice);

	mat d_C = {  .N = C.N, .M = C.M, .step = C.M};
	size = C.M * C.N * sizeof(float);
	cudaMalloc(&d_C.m, size);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.M / dimBlock.x, A.N / dimBlock.y);
	matmul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	cudaMemcpy(C.m, d_C.m, size, cudaMemcpyDeviceToHost);
	
	int N = C.N, M = C.M;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			float elem = C.m[i * C.step + j];
			printf("%f ", elem);
		}
		printf("\n");
	}

	cudaFree(d_A.m);
	cudaFree(d_B.m);
	cudaFree(d_C.m);
	free(m3);
}

