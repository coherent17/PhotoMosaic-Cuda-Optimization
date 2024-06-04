#include <stdio.h>

#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512

__global__ void add(int *a, int *b, int *c){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

int main(){
	int *a, *b, *c;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;	// device copies of a, b, c
	int size = N * sizeof(int);

	// allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// allocate space for host copies of a, b, c
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	for(int i = 0; i < N; i++){
		a[i] = i;
		b[i] = i;
	}

	// copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// launch add kernel on GPU with N threads
	add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

	// copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++){
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	// cleanup
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
