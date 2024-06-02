#include "photo_mosaic_serial.h"

Data_Loader Photo_Mosaic_Serial::data_loader;

Photo_Mosaic_Serial::Photo_Mosaic_Serial(){
    this->num_candidate_imgs = 0;
}

Photo_Mosaic_Serial::~Photo_Mosaic_Serial(){

}

RGBImage *Photo_Mosaic_Serial::Run(string targetImgPath, string candidateImgFolderPath){
    target_img.LoadImage(targetImgPath);
    target_img.Display_CMD();

    RGBImage* result = new RGBImage(target_img.get_w(), target_img.get_h());
    data_loader.List_Directory(candidateImgFolderPath, candidate_img_filenames);
    RGBImage candidate_imgs[NUM_CANDIDATE_IMGS];

    for(const auto filename : candidate_img_filenames){
        if(candidate_imgs[num_candidate_imgs].LoadImage(filename)){
            num_candidate_imgs++;
        }
    }

    for(int row = 0; row < target_img.get_h() - SUB_PIC_SIZE; row += SUB_PIC_SIZE){
        for(int col = 0; col < target_img.get_w() - SUB_PIC_SIZE; col += SUB_PIC_SIZE){
            // calculate the r_avg, g_avg, b_avg of the grid of the target image
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

            // find the best fit candidate images
            int best_candidate_index = 0;
            double min_diff = DBL_MAX;
            for(int i = 0; i < num_candidate_imgs; i++){
                double r_diff = r_avg_target - candidate_imgs[i].r_avg;
                double g_diff = g_avg_target - candidate_imgs[i].g_avg;
                double b_diff = b_avg_target - candidate_imgs[i].b_avg;
                double diff = sqrt(r_diff * r_diff + g_diff * g_diff + b_diff * b_diff);
                if(diff < min_diff){
                    best_candidate_index = i;
                    min_diff = diff;
                }
            }

            // replace the target image with candidate image
            for(int i = row; i < row + SUB_PIC_SIZE; i++){
                for(int j = col; j < col + SUB_PIC_SIZE; j++){
                    result->pixels[i][j][0] = candidate_imgs[best_candidate_index].pixels[i-row][j-col][0];
                    result->pixels[i][j][1] = candidate_imgs[best_candidate_index].pixels[i-row][j-col][1];
                    result->pixels[i][j][2] = candidate_imgs[best_candidate_index].pixels[i-row][j-col][2];
                }
            }
        }
    }

    return result;
}