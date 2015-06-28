#include "../common/book.h"

#define N 10

__global__ void add(int *a, int *b, int *c) {
	int tid = threadIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main(void) {
	int h_a[N], h_b[N], h_c[N];
	int *d_a, *d_b, *d_c;

	for (int i = 0; i < N; i++)
	{
		h_a[i] = i;
		h_b[i] = i * i;
	}

	HANDLE_ERROR(cudaMalloc((void**)&d_a, N * sizeof(int)));	
	HANDLE_ERROR(cudaMalloc((void**)&d_b, N* sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_c, N* sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice));
	add << <1, N >> >(d_a, d_b, d_c);

	HANDLE_ERROR(cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++)
	{
		printf("results of a + b = c:\n%d + %d = %d\n\n", h_a[i], h_b[i], h_c[i]);
	}
	return 0;

	HANDLE_ERROR(cudaFree(d_a));
	HANDLE_ERROR(cudaFree(d_b));
	HANDLE_ERROR(cudaFree(d_c));
};