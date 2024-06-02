#ifndef _PHOTO_MOSAIC_CUDA_H_
#define _PHOTO_MOSAIC_CUDA_H_

#define SUB_PIC_SIZE 32
#define NUM_CANDIDATE_IMGS 10000
#include "rgb_image.h"

class Photo_Mosaic_CUDA {
private:
    static Data_Loader data_loader;
    RGBImage target_img;
    int num_candidate_imgs;
    std::vector<std::string> candidate_img_filenames;
    RGBImage *candidate_imgs;

public:
    Photo_Mosaic_CUDA();
    ~Photo_Mosaic_CUDA();
    RGBImage *Run(const std::string &targetImgPath, const std::string &candidateImgFolderPath);
};

#endif
