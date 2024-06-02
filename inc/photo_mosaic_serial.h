#ifndef _PHOTO_MOSAIC_SERIAL_H_
#define _PHOTO_MOSAIC_SERIAL_H_

#define SUB_PIC_SIZE 32
#define NUM_CANDIDATE_IMGS 10000
#include "rgb_image.h"

class Photo_Mosaic_Serial{
private:
    static Data_Loader data_loader;
    RGBImage target_img;
    int num_candidate_imgs;
    vector<string> candidate_img_filenames;

public:
    Photo_Mosaic_Serial();
    ~Photo_Mosaic_Serial();
    RGBImage *Run(string targetImgPath, string candidateImgFolderPath);
};

#endif