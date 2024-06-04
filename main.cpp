#include "data_loader.h"
#include "photo_mosaic_serial.h"
#include "photo_mosaic_cuda.cuh"

int main(){
    Photo_Mosaic_Serial photo_mosaic_serial;
    RGBImage* result1 = photo_mosaic_serial.Run("Image-Folder/girl_2x.png","Image-Folder/cifar10");
    result1->DumpImage("img_serial.jpg");
    result1->Display_CMD();

    Photo_Mosaic_Cuda photo_mosaic_cuda;
    RGBImage* result2 = photo_mosaic_cuda.Run("Image-Folder/girl_2x.png","Image-Folder/cifar10");
    result2->DumpImage("img_cuda.jpg");
    result2->Display_CMD();

    delete result1;
    delete result2;
    return 0;
}
