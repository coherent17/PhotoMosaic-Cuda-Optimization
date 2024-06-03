#include <driver_types.h>
#include <stdio.h>

// both a, b, c are the pointer on the device memory
// should allocate memory on the device first!
__global__ void add(int *a, int *b, int *c){
	*c = *a + *b;
}

int main(){
	int a, b, c;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;	// device copies of a, b, c
	int size = sizeof(int);

	// allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// setup input values
	a = 2;
	b = 7;

	// copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	// launch add kernel on GPU
	add<<<1, 1>>>(d_a, d_b, d_c);

	// copy result back to host
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	// cleanup
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("c = %d\n", c);
	return 0;
}
