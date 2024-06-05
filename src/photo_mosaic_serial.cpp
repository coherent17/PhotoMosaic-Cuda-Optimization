#include "photo_mosaic_serial.h"

Data_Loader Photo_Mosaic_Serial::data_loader;

Photo_Mosaic_Serial::Photo_Mosaic_Serial(){
    this->num_candidate_imgs = 0;
}

Photo_Mosaic_Serial::~Photo_Mosaic_Serial(){
    free(r_avg_candidate);
    free(g_avg_candidate);
    free(b_avg_candidate);
    free(r_avg_target_grid);
    free(g_avg_target_grid);
    free(b_avg_target_grid);
    free(min_idxs);
}

RGBImage *Photo_Mosaic_Serial::Run(string targetImgPath, string candidateImgFolderPath){
    target_img.LoadImage(targetImgPath);
    RGBImage* result = new RGBImage(target_img.get_w(), target_img.get_h());
    data_loader.List_Directory(candidateImgFolderPath, candidate_img_filenames);
    RGBImage candidate_imgs[NUM_CANDIDATE_IMGS];

    for(const auto filename : candidate_img_filenames){
        if(candidate_imgs[num_candidate_imgs].LoadImage(filename)){
            num_candidate_imgs++;
        }
    }

    // allocate memory [target image]
    int tile_width = target_img.get_w() / SUB_PIC_SIZE;
    int tile_height = target_img.get_h() / SUB_PIC_SIZE;
    r_avg_target_grid = (double *)malloc(sizeof(double) * tile_width * tile_height);
    g_avg_target_grid = (double *)malloc(sizeof(double) * tile_width * tile_height);
    b_avg_target_grid = (double *)malloc(sizeof(double) * tile_width * tile_height);
    min_idxs = (int *)malloc(sizeof(int) * tile_width * tile_height);

    // allocate memory [candidate image]
    r_avg_candidate = (double *)malloc(sizeof(double) * num_candidate_imgs);
    g_avg_candidate = (double *)malloc(sizeof(double) * num_candidate_imgs);
    b_avg_candidate = (double *)malloc(sizeof(double) * num_candidate_imgs);

    // calculate the r_avg, g_avg, b_avg of the grid of the target image
    int target_grid_idx = 0;
    for(int row = 0; row < target_img.get_h() - SUB_PIC_SIZE; row += SUB_PIC_SIZE){
        for(int col = 0; col < target_img.get_w() - SUB_PIC_SIZE; col += SUB_PIC_SIZE){
            double r_avg_target = 0;
            double g_avg_target = 0;
            double b_avg_target = 0;
            for(int i = row; i < row + SUB_PIC_SIZE; i++){
                for(int j = col; j < col + SUB_PIC_SIZE; j++){
                    r_avg_target += target_img.pixels[i][j][0];
                    g_avg_target += target_img.pixels[i][j][1];
                    b_avg_target += target_img.pixels[i][j][2];
                }
            }
            r_avg_target /= SUB_PIC_SIZE * SUB_PIC_SIZE;
            g_avg_target /= SUB_PIC_SIZE * SUB_PIC_SIZE;
            b_avg_target /= SUB_PIC_SIZE * SUB_PIC_SIZE;
            r_avg_target_grid[target_grid_idx] = r_avg_target;
            g_avg_target_grid[target_grid_idx] = g_avg_target;
            b_avg_target_grid[target_grid_idx] = b_avg_target;
            target_grid_idx++;
        }
    }

    for(int i = 0; i < num_candidate_imgs; i++){
        r_avg_candidate[i] = candidate_imgs[i].r_avg;
        g_avg_candidate[i] = candidate_imgs[i].g_avg;
        b_avg_candidate[i] = candidate_imgs[i].b_avg;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int i = 0; i < tile_width * tile_height; i++){
        double min_diff = DBL_MAX;
        for(int j= 0; j < num_candidate_imgs; j++){
            double r_diff = r_avg_target_grid[i] - r_avg_candidate[j];
            double g_diff = g_avg_target_grid[i] - g_avg_candidate[j];
            double b_diff = b_avg_target_grid[i] - b_avg_candidate[j];
            double diff = sqrt(r_diff * r_diff + g_diff * g_diff + b_diff * b_diff);
            if(diff < min_diff){
                min_idxs[i] = j;
                min_diff = diff;
            }
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("[Serial]Elapsed time: %f ms\n", elapsedTime);
    
    int count = 0;
    for(int row = 0; row < target_img.get_h() - SUB_PIC_SIZE; row += SUB_PIC_SIZE){
        for(int col = 0; col < target_img.get_w() - SUB_PIC_SIZE; col += SUB_PIC_SIZE){
            for(int i = row; i < row + SUB_PIC_SIZE; i++){
                for(int j = col; j < col + SUB_PIC_SIZE; j++){
                    result->pixels[i][j][0] = candidate_imgs[min_idxs[count]].pixels[i-row][j-col][0];
                    result->pixels[i][j][1] = candidate_imgs[min_idxs[count]].pixels[i-row][j-col][1];
                    result->pixels[i][j][2] = candidate_imgs[min_idxs[count]].pixels[i-row][j-col][2];        
                }
            }
            count++;
        }
    }

    return result;
}