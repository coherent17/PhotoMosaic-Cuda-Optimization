#include "data_loader.h"
#include "photo_mosaic_serial.h"
#include "photo_mosaic_cuda.cuh"

int main(){

	Photo_Mosaic_Serial photo_mosaic_serial;
	Photo_Mosaic_Cuda photo_mosaic_cuda;
	RGBImage *result_serial, *result_cuda;

	// test case1: owl
	result_serial = photo_mosaic_serial.Run("Image-Folder/4k_owl.jpg","Image-Folder/cifar10");
	result_serial->DumpImage("4k_owl_serial.jpg");
	result_serial->Display_CMD();
	result_cuda = photo_mosaic_cuda.Run("Image-Folder/4k_owl.jpg","Image-Folder/cifar10");
	result_cuda->DumpImage("4k_owl_cuda.jpg");
	result_cuda->Display_CMD();
	delete result_serial;
	delete result_cuda;

	return 0;
}
