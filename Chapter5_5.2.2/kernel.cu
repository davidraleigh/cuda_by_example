#include "../common/book.h"
#include "../common/cpu_anim.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024 / 2

struct DataBlock {
	unsigned char *d_bitmap;
	CPUAnimBitmap *h_bitmap;
};

void cleanup(DataBlock *d) {
	cudaFree(d->d_bitmap);
}

__global__ void kernel(unsigned char *ptr, int ticks) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);

	unsigned char grey = (unsigned char)(128.0f + 127.0f * 
										 cos(d / 10.0f - ticks / 7.0f) / 
										 (d / 10.0f + 1.0f));
	ptr[offset * 4 + 0] = grey;
	ptr[offset * 4 + 1] = grey;
	ptr[offset * 4 + 2] = grey;
	ptr[offset * 4 + 3] = 255;
}

void generate_frame(DataBlock *d, int ticks) {
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel<<<blocks, threads>>>(d->d_bitmap, ticks);

	HANDLE_ERROR(cudaMemcpy(d->h_bitmap->get_ptr(),
							d->d_bitmap,
							d->h_bitmap->image_size(),
							cudaMemcpyDeviceToHost));
}

int main(void) {
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.h_bitmap = &bitmap;

	HANDLE_ERROR(cudaMalloc((void**)&data.d_bitmap, bitmap.image_size()));

	bitmap.anim_and_exit( (void (*)(void*,int))generate_frame, 
						  (void (*)(void*))cleanup);
	return 0;
}
