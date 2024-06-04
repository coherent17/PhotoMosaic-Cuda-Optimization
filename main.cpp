#include "data_loader.h"
#include "photo_mosaic_serial.h"
#include "photo_mosaic_cuda.cuh"

int main(){
  	cudaEvent_t start_s, stop_s;
	cudaEventCreate (&start_s);
	cudaEventCreate (&stop_s); 	

    cudaEventRecord(start_s, 0);
    Photo_Mosaic_Serial photo_mosaic_serial;
    RGBImage* result1 = photo_mosaic_serial.Run("Image-Folder/4k_owl.jpg","Image-Folder/cifar10");
    cudaEventRecord(stop_s, 0);
    result1->DumpImage("img_serial.jpg");
    result1->Display_CMD();
    float elapsedTime_s; 
  	cudaEventElapsedTime(&elapsedTime_s, start_s, stop_s);
    printf("%5.2f ms\n", elapsedTime_s);


    cudaEvent_t start_c, stop_c;
	cudaEventCreate (&start_c);
	cudaEventCreate (&stop_c);

    cudaEventRecord(start_c, 0);
    Photo_Mosaic_Cuda photo_mosaic_cuda;
    RGBImage* result2 = photo_mosaic_cuda.Run("Image-Folder/4k_owl.jpg","Image-Folder/cifar10");
    cudaEventRecord(stop_c, 0);
    result2->DumpImage("img_cuda.jpg");
    result2->Display_CMD();
    float elapsedTime_c; 
  	cudaEventElapsedTime(&elapsedTime_c, start_c, stop_c);
    printf("%5.2f ms\n", elapsedTime_c);

    delete result1;
    delete result2;
    return 0;
}
