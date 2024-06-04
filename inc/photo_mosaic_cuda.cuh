#ifndef _PHOTO_MOSAIC_CUDA_H_
#define _PHOTO_MOSAIC_CUDA_H_

#define SUB_PIC_SIZE 32
#define NUM_CANDIDATE_IMGS 10000
#include "rgb_image.h"

class Photo_Mosaic_Cuda{
private:
    static Data_Loader data_loader;
    RGBImage target_img;
    int num_candidate_imgs;
    vector<string> candidate_img_filenames;
    double *r_avg_candidate;
    double *g_avg_candidate;
    double *b_avg_candidate;
    double *r_avg_target_grid;
    double *g_avg_target_grid;
    double *b_avg_target_grid;
    int *min_idxs;

public:
    Photo_Mosaic_Cuda();
    ~Photo_Mosaic_Cuda();
    RGBImage *Run(string targetImgPath, string candidateImgFolderPath);
};

#endif