#include "../common/book.h"
#include <vector>

#define N (66 * 1024)

__global__ void addRobust(int *a, int *b, int *c) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];

		// gives the number of threads in a grid (in the x direction, in this case)
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void add(int *a, int *b, int *c) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// This line is now more understandable. as a result of over estimating
	// the number of threads necessary beause of block size we need to skip the 
	// cases where the thread index is outside of the size of the problem's limits
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

bool checkData(int *a, int *b, int *c) {
	bool success = true;
	for (int i = 0; i < N; i++)
	{
		if ((a[i] + b[i]) != c[i]) {
			printf("Error at index : %d\n", i);
			success = false;
		}
	}
	if (success) {
		printf("successfully calculated at all indices\n");
	}
	else {
		printf("Failed at calculating at all indices\n");
	}
	return success;
}

int main(void) {
	printf("Calculations to make : %d\n", N);

	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
	printf("Max threads per block : %d\n", prop.maxThreadsPerBlock);
	printf("Max grid dim : %d\n", prop.maxGridSize[0]);
	int blocksNeeded = (N + (prop.maxThreadsPerBlock - 1)) / prop.maxThreadsPerBlock;
	printf("Blocks needed to complete computation : %d\n", blocksNeeded);

	size_t free_byte;
	size_t total_byte;
	HANDLE_ERROR(cudaMemGetInfo(&free_byte, &total_byte));

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("Free memory %f\n", free_db);

	int h_a[N], h_b[N], h_c[N];
	for (int i = 0; i < N; i++)
	{
		h_a[i] = i;
		h_b[i] = i * i;
	}

	int *d_a, *d_b, *d_c, *d_d;
	HANDLE_ERROR(cudaMalloc((void**)&d_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_c, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_d, N * sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice));

	add <<<N + (prop.maxThreadsPerBlock - 1), prop.maxThreadsPerBlock >>>(d_a, d_b, d_c);
	HANDLE_ERROR(cudaMemGetInfo(&free_byte, &total_byte));

	free_db = (double)free_byte;
	total_db = (double)total_byte;
	used_db = total_db - free_db;
	printf("Free memory %f\n", free_db);

	HANDLE_ERROR(cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost));
	checkData(h_a, h_b, h_c);

	addRobust<<<prop.maxThreadsPerBlock, prop.maxThreadsPerBlock>>>(d_a, d_b, d_d);
	HANDLE_ERROR(cudaMemcpy(h_c, d_d, N*sizeof(int), cudaMemcpyDeviceToHost));
	checkData(h_a, h_b, h_c);

	return 0;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

