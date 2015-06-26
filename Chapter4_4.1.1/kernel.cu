#include "../common/book.h"

#define N 10

__global__ void cudaAdd(int *a, int *b, int *c) {
	int tid = blockIdx.x;// get index of this kernel's call
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

void add(int *a, int *b, int *c) {
	int tid = 0;
	while (tid < 10) {
		c[tid] = a[tid] + b[tid];
		tid += 1;
	}
}

int main(void) {
	int a[N], b[N], c[N];

	// set numbers
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	add(a, b, c);

	// display results
	for (int i = 0; i < N; i++) {
		printf("CPU results: %d + %d = %d\n", a[i], b[i], c[i]);
	}

	// CUDA version
	int h_a[N], h_b[N], h_c[N];
	int *d_a, *d_b, *d_c;

	HANDLE_ERROR(cudaMalloc((void**)&d_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_c, N * sizeof(int)));

	for (int i = 0; i < N; i++) {
		h_a[i] = -i;
		h_b[i] = i * i;
	}

	HANDLE_ERROR(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice));

	cudaAdd <<<N, 1 >>>(d_a, d_b, d_c);

	HANDLE_ERROR(cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost));
	
	// display results
	for (int i = 0; i < N; i++) {
		printf("GPU results: %d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return 0;
}