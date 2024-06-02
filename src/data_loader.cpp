#include "data_loader.h"

Data_Loader::Data_Loader(){
    ;
}

// Assume user will free the image
Data_Loader::~Data_Loader(){
    ;
}

int **Data_Loader::Load_Gray(string filename, int *w, int *h){
    assert(File_Exists(filename));
    CImg<unsigned char> img(filename.c_str());

    int _w = img.width();
    int _h = img.height();
    int _c = img.spectrum();

    *w = _w;
    *h = _h;

    // allocate memory for the image array
    int **pixels = new int *[_h];
    for(int i = 0; i < _h; i++){
        pixels[i] = new int[_w];
    }

    // gray scale image
    if(_c == 1){
        // macro to loop through the img
        cimg_forXY(img, x, y){
            pixels[y][x] = (int)img(x,y);
        }
        return pixels;
    }
    
    // rgb img -> convert it into gray scale img
    if(_c == 3){
        CImg<unsigned char> grayscale_img(_w, _h, 1);
        cimg_forXY(grayscale_img, x, y){
            grayscale_img(x, y) = (unsigned char)(R_FACTOR * img(x, y, 0, 0) + G_FACTOR * img(x, y, 0, 1) + B_FACTOR * img(x, y, 0, 2));
        }
        cimg_forXY(img, x, y){
            pixels[y][x] = (int)grayscale_img(x,y);
        }
        return pixels;
    }

    // rgba image -> convert it into gray scale image
    if(_c == 4){

        // Allocate memory for the 3D array
        int ***pixel3 = new int**[_h];
        for(int i = 0; i < _h; ++i) {
            pixel3[i] = new int*[_w];
            for(int j = 0; j < _w; ++j) {
                pixel3[i][j] = new int[3];
            }
        }

        // Copy pixel values from the image to the 3D array
        cimg_forXYC(img, x, y, c) {
            if(c < 3){
                pixel3[y][x][c] = img(x, y, c);
            }
        }

        for(int i = 0; i < _h; i++){
            for(int j = 0; j < _w; j++){
                pixels[i][j] = R_FACTOR * pixel3[i][j][0] + G_FACTOR * pixel3[i][j][1] + B_FACTOR * pixel3[i][j][2];
            }
        }
        for(int i = 0; i < _h; i++){
            for(int j = 0; j < _w; j++){
                delete []pixel3[i][j];
            }
            delete []pixel3[i];
        }
        delete []pixel3;
        return pixels;
    }

    return nullptr;
}

int ***Data_Loader::Load_RGB(string filename, int *w, int *h){
    assert(File_Exists(filename));
    CImg<unsigned char> img(filename.c_str());

    int _w = img.width();
    int _h = img.height();
    int _c = img.spectrum();

    *w = _w;
    *h = _h;

    if(_c < 3) return nullptr;

    // Allocate memory for the 3D array
    int ***pixels = new int**[_h];
    for(int i = 0; i < _h; ++i) {
        pixels[i] = new int*[_w];
        for(int j = 0; j < _w; ++j) {
            pixels[i][j] = new int[3];
        }
    }

    // Copy pixel values from the image to the 3D array
    cimg_forXYC(img, x, y, c) {
        if(c < 3){
            pixels[y][x][c] = img(x, y, c);
        }
    }
    return pixels;
}

void Data_Loader::Dump_Gray(int w, int h, int **pixels, string filename){
    assert(pixels != nullptr && w > 0 && h > 0);
    // Create a CImg object with the specified width and height
    CImg<unsigned char> img(w, h, 1, 1); // Grayscale image (1 channel)

    // Iterate through the image data and set pixel values
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Set the grayscale pixel value
            img(x, y) = (unsigned char)pixels[y][x];
        }
    }
    img.save(filename.c_str());

}

void Data_Loader::Dump_RGB(int w, int h, int ***pixels, string filename){
    assert(pixels != nullptr && w > 0 && h > 0);
    // Create a CImg object with the specified width, height, and 3 channels (RGB)
    CImg<unsigned char> img(w, h, 1, 3);

    // Iterate through the pixel values and set them in the CImg object
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Set the RGB pixel values
            img(x, y, 0) = pixels[y][x][0]; // Red channel
            img(x, y, 1) = pixels[y][x][1]; // Green channel
            img(x, y, 2) = pixels[y][x][2]; // Blue channel
        }
    }

    // Save the image to a file
    img.save(filename.c_str());
}

void Data_Loader::Display_Gray_CMD(string filename){
    assert(File_Exists(filename));
    string cmd = "./third-party/catimg/bin/catimg " + string(filename);
    system(cmd.c_str());
}

void Data_Loader::Display_RGB_CMD(string filename){
    assert(File_Exists(filename));
    string cmd = "./third-party/catimg/bin/catimg " + string(filename);
    system(cmd.c_str());
}

bool Data_Loader::List_Directory(string directoryPath, vector<string> &filenames) {
    struct dirent *entry;
    DIR *dp;

    dp = opendir(directoryPath.c_str());
    if (dp == NULL) {
        perror("opendir: Path does not exist or could not be read.");
        return -1;
    }

    while ((entry = readdir(dp))){
        // store all the .png filename into vector
        if(string(entry->d_name) == "." || string(entry->d_name) == "..") continue;
        filenames.push_back(directoryPath + "/" + string(entry->d_name));
    }

    closedir(dp);
    return 0;
}

bool Data_Loader::File_Exists(const string &filename){
    ifstream file(filename.c_str());
    return file.good();
}
