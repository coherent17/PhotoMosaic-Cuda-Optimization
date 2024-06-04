#include "data_loader.h"
#include "photo_mosaic_serial.h"
#include "photo_mosaic_cuda.cuh"

int main(){
    // Data_Loader data_loader;
    // int w3;
    // int h3;
    // int ***pixels3 = data_loader.Load_RGB("Image-Folder/cifar10/airplane_0010.png", &w3, &h3);
    // data_loader.Dump_RGB(w3, h3, pixels3, string("pixels3.jpg"));
    // data_loader.Display_RGB_CMD("pixels3.jpg");

    Photo_Mosaic_Serial photo_mosaic_serial;
    RGBImage* result1 = photo_mosaic_serial.Run("Image-Folder/girl_2x.png","Image-Folder/cifar10");
    result1->DumpImage("img3.jpg");
    result1->Display_CMD();

    Photo_Mosaic_CUDA photo_mosaic_cuda;
    RGBImage* result2 = photo_mosaic_cuda.Run("Image-Folder/girl_2x.png","Image-Folder/cifar10");
    result2->DumpImage("img3.jpg");
    result2->Display_CMD();
    return 0;
}
