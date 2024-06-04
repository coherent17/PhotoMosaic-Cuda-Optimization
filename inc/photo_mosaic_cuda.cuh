#ifndef _PHOTO_MOSAIC_CUDA_H_
#define _PHOTO_MOSAIC_CUDA_H_

#include "rgb_image.h"
#include <cuda_runtime.h>

#define SUB_PIC_SIZE 32
#define NUM_CANDIDATE_IMGS 10000
#define THREADS_PER_BLOCK 512
#define IMAGE_DBL_MAX 300

class Photo_Mosaic_Cuda{
private:
    static Data_Loader data_loader;
    RGBImage target_img;
    int num_candidate_imgs;
    vector<string> candidate_img_filenames;

    // host memory
    double *r_avg_candidate;
    double *g_avg_candidate;
    double *b_avg_candidate;
    double *r_avg_target_grid;
    double *g_avg_target_grid;
    double *b_avg_target_grid;
    int *min_idxs;

    // device memory
    double *d_r_avg_target_grid;
    double *d_g_avg_target_grid;
    double *d_b_avg_target_grid;
    double *d_r_avg_candidate;
    double *d_g_avg_candidate;
    double *d_b_avg_candidate;
    int *d_min_idxs;

public:
    Photo_Mosaic_Cuda();
    ~Photo_Mosaic_Cuda();
    RGBImage *Run(string targetImgPath, string candidateImgFolderPath);
};

#endif