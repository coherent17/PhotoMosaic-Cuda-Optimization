#include <stdio.h>
#define N 512

__global__ void add(int *a, int *b, int *c){
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
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
	add<<<1, N>>>(d_a, d_b, d_c);

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
