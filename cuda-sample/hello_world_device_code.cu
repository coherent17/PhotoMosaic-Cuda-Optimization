#include <stdio.h>

// all kernel has void return type, copy from PCI through pointer
// __global__: This keyword specifies that the function is a kernel function. 
// Kernel functions are executed on the GPU, and they can be called from the host (CPU) code.

__global__ void mykernel(void){

}

int main(){
	// run it on the 1 block and the 1 thread
	mykernel<<<1, 1>>>();
	printf("Hello World!\n");
	return 0;
}
