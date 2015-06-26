#include "../common/book.h"

int main()
{
	cudaDeviceProp prop;

	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) 
	{
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
		printf(" Device # %d \n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Total Global memory %ld\n", prop.totalGlobalMem);

		printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions (%d %d %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	}

    return 0;
}