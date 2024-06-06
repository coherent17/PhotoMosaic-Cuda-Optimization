#include "data_loader.h"
#include "photo_mosaic_serial.h"
#include "photo_mosaic_cuda.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define DUMP_IMAGE true
#define CMD_IMAGE true

void checkCudaDevices() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA-compatible devices found." << std::endl;
        return;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    }
}

int main(){

	checkCudaDevices();

	Photo_Mosaic_Serial *photo_mosaic_serial;
	Photo_Mosaic_Cuda *photo_mosaic_cuda;
	RGBImage *result_serial, *result_cuda;

	// test case1: owl
	printf("################## Test case1 ##################\n");
	photo_mosaic_serial = new Photo_Mosaic_Serial();
	result_serial = photo_mosaic_serial->Run("Image-Folder/4k_owl.jpg","Image-Folder/cifar10");
	if(DUMP_IMAGE) result_serial->DumpImage("4k_owl_serial.jpg");
	if(CMD_IMAGE) result_serial->Display_CMD();
	photo_mosaic_cuda = new Photo_Mosaic_Cuda();
	result_cuda = photo_mosaic_cuda->Run("Image-Folder/4k_owl.jpg","Image-Folder/cifar10");
	if(DUMP_IMAGE) result_cuda->DumpImage("4k_owl_cuda.jpg");
	if(CMD_IMAGE) result_cuda->Display_CMD();
	delete result_serial;
	delete result_cuda;
	delete photo_mosaic_serial;
	delete photo_mosaic_cuda;

	// test case2: sunflower
	printf("################## Test case2 ##################\n");
	photo_mosaic_serial = new Photo_Mosaic_Serial();
	result_serial = photo_mosaic_serial->Run("Image-Folder/sunflower.jpg","Image-Folder/cifar10");
	if(DUMP_IMAGE) result_serial->DumpImage("sunflower_serial.jpg");
	if(CMD_IMAGE) result_serial->Display_CMD();
	photo_mosaic_cuda = new Photo_Mosaic_Cuda();
	result_cuda = photo_mosaic_cuda->Run("Image-Folder/sunflower.jpg","Image-Folder/cifar10");
	if(DUMP_IMAGE) result_cuda->DumpImage("sunflower_cuda.jpg");
	if(CMD_IMAGE) result_cuda->Display_CMD();
	delete result_serial;
	delete result_cuda;
	delete photo_mosaic_serial;
	delete photo_mosaic_cuda;

	// test case3: dogs
	printf("################## Test case3 ##################\n");
	photo_mosaic_serial = new Photo_Mosaic_Serial();
	result_serial = photo_mosaic_serial->Run("Image-Folder/dog.jpg","Image-Folder/cifar10");
	if(DUMP_IMAGE) result_serial->DumpImage("dog_serial.jpg");
	if(CMD_IMAGE) result_serial->Display_CMD();
	photo_mosaic_cuda = new Photo_Mosaic_Cuda();
	result_cuda = photo_mosaic_cuda->Run("Image-Folder/dog.jpg","Image-Folder/cifar10");
	if(DUMP_IMAGE) result_cuda->DumpImage("dog_cuda.jpg");
	if(CMD_IMAGE) result_cuda->Display_CMD();
	delete result_serial;
	delete result_cuda;
	delete photo_mosaic_serial;
	delete photo_mosaic_cuda;

	// test case4: arc
	printf("################## Test case4 ##################\n");
	photo_mosaic_serial = new Photo_Mosaic_Serial();
	result_serial = photo_mosaic_serial->Run("Image-Folder/arc.jpg","Image-Folder/cifar10");
	if(DUMP_IMAGE) result_serial->DumpImage("arc_serial.jpg");
	if(CMD_IMAGE) result_serial->Display_CMD();
	photo_mosaic_cuda = new Photo_Mosaic_Cuda();
	result_cuda = photo_mosaic_cuda->Run("Image-Folder/arc.jpg","Image-Folder/cifar10");
	if(DUMP_IMAGE) result_cuda->DumpImage("arc_cuda.jpg");
	if(CMD_IMAGE) result_cuda->Display_CMD();
	delete result_serial;
	delete result_cuda;
	delete photo_mosaic_serial;
	delete photo_mosaic_cuda;

	return 0;
}
