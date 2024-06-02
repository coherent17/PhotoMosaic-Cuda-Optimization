#include "data_loader.h"

int main(){
    Data_Loader data_loader;
    int w3;
    int h3;
    int ***pixels3 = data_loader.Load_RGB("Image-Folder/cifar10/airplane_0010.png", &w3, &h3);
    data_loader.Dump_RGB(w3, h3, pixels3, string("pixels3.jpg"));
    data_loader.Display_RGB_CMD("pixels3.jpg");
    return 0;
}