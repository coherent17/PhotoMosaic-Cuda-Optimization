#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "photo_mosaic_cuda.cuh"
#include <cfloat>

Data_Loader Photo_Mosaic_CUDA::data_loader;

Photo_Mosaic_CUDA::Photo_Mosaic_CUDA() {
    this->num_candidate_imgs = 0;
    candidate_imgs = new RGBImage[NUM_CANDIDATE_IMGS];
}

Photo_Mosaic_CUDA::~Photo_Mosaic_CUDA() {
    delete[] candidate_imgs;
}

// Custom atomicAdd for double
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Custom atomicMin for double
__device__ double atomicMin(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(min(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ int atomicExch(int* address, int val) {
    return atomicCAS(address, *address, val);
}

__global__ void calculate_averages_kernel(RGBImage target_img, int start_row, int start_col, double* r_avg_target, double* g_avg_target, double* b_avg_target) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + start_row;
    int col = blockIdx.x * blockDim.x + threadIdx.x + start_col;

    if (row < start_row + SUB_PIC_SIZE && col < start_col + SUB_PIC_SIZE) {
        atomicAdd(r_avg_target, (double)target_img.pixels[row][col][0]);
        atomicAdd(g_avg_target, (double)target_img.pixels[row][col][1]);
        atomicAdd(b_avg_target, (double)target_img.pixels[row][col][2]);
    }
}

__global__ void find_best_fit_kernel(RGBImage* candidate_imgs, int num_candidate_imgs, double r_avg_target, double g_avg_target, double b_avg_target, int* best_candidate_index, double* min_diff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_candidate_imgs) {
        double r_diff = r_avg_target - candidate_imgs[idx].r_avg;
        double g_diff = g_avg_target - candidate_imgs[idx].g_avg;
        double b_diff = b_avg_target - candidate_imgs[idx].b_avg;
        double diff = sqrt(r_diff * r_diff + g_diff * g_diff + b_diff * b_diff);

        double old_min_diff = atomicMin(min_diff, diff);
        if (diff < old_min_diff) {
            atomicExch(best_candidate_index, idx);
        }
    }
}

__global__ void replace_image_kernel(RGBImage target_img, RGBImage* result_img, RGBImage best_candidate_img, int start_row, int start_col) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + start_row;
    int col = blockIdx.x * blockDim.x + threadIdx.x + start_col;

    if (row < start_row + SUB_PIC_SIZE && col < start_col + SUB_PIC_SIZE) {
        result_img->pixels[row][col][0] = best_candidate_img.pixels[row - start_row][col - start_col][0];
        result_img->pixels[row][col][1] = best_candidate_img.pixels[row - start_row][col - start_col][1];
        result_img->pixels[row][col][2] = best_candidate_img.pixels[row - start_row][col - start_col][2];
    }
}

RGBImage* Photo_Mosaic_CUDA::Run(const std::string &targetImgPath, const std::string &candidateImgFolderPath) {
    target_img.LoadImage(targetImgPath);
    target_img.Display_CMD();

    RGBImage* result = new RGBImage(target_img.get_w(), target_img.get_h());
    data_loader.List_Directory(candidateImgFolderPath, candidate_img_filenames);

    for (const auto& filename : candidate_img_filenames) {
        if (candidate_imgs[num_candidate_imgs].LoadImage(filename)) {
            num_candidate_imgs++;
        }
    }

    // Copy data to device
    RGBImage* d_candidate_imgs;
    cudaMalloc(&d_candidate_imgs, num_candidate_imgs * sizeof(RGBImage));
    cudaMemcpy(d_candidate_imgs, candidate_imgs, num_candidate_imgs * sizeof(RGBImage), cudaMemcpyHostToDevice);

    RGBImage* d_target_img;
    cudaMalloc(&d_target_img, sizeof(RGBImage));
    cudaMemcpy(d_target_img, &target_img, sizeof(RGBImage), cudaMemcpyHostToDevice);

    RGBImage* d_result_img;
    cudaMalloc(&d_result_img, sizeof(RGBImage));
    cudaMemcpy(d_result_img, result, sizeof(RGBImage), cudaMemcpyHostToDevice);

    dim3 blockDim(SUB_PIC_SIZE, SUB_PIC_SIZE);
    dim3 gridDim((target_img.get_w() + blockDim.x - 1) / blockDim.x, (target_img.get_h() + blockDim.y - 1) / blockDim.y);

    for (int row = 0; row < target_img.get_h() - SUB_PIC_SIZE; row += SUB_PIC_SIZE) {
        for (int col = 0; col < target_img.get_w() - SUB_PIC_SIZE; col += SUB_PIC_SIZE) {
            double r_avg_target = 0, g_avg_target = 0, b_avg_target = 0;
            double* d_r_avg_target;
            double* d_g_avg_target;
            double* d_b_avg_target;
            cudaMalloc(&d_r_avg_target, sizeof(double));
            cudaMalloc(&d_g_avg_target, sizeof(double));
            cudaMalloc(&d_b_avg_target, sizeof(double));
            cudaMemcpy(d_r_avg_target, &r_avg_target, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_g_avg_target, &g_avg_target, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b_avg_target, &b_avg_target, sizeof(double), cudaMemcpyHostToDevice);

            calculate_averages_kernel<<<gridDim, blockDim>>>(*d_target_img, row, col, d_r_avg_target, d_g_avg_target, d_b_avg_target);
            cudaMemcpy(&r_avg_target, d_r_avg_target, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&g_avg_target, d_g_avg_target, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&b_avg_target, d_b_avg_target, sizeof(double), cudaMemcpyDeviceToHost);

            cudaFree(d_r_avg_target);
            cudaFree(d_g_avg_target);
            cudaFree(d_b_avg_target);

            int best_candidate_index = 0;
            double min_diff = DBL_MAX;
            int* d_best_candidate_index;
            double* d_min_diff;
            cudaMalloc(&d_best_candidate_index, sizeof(int));
            cudaMalloc(&d_min_diff, sizeof(double));
            cudaMemcpy(d_best_candidate_index, &best_candidate_index, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_min_diff, &min_diff, sizeof(double), cudaMemcpyHostToDevice);

            find_best_fit_kernel<<<(num_candidate_imgs + blockDim.x - 1) / blockDim.x, blockDim.x>>>(d_candidate_imgs, num_candidate_imgs, r_avg_target, g_avg_target, b_avg_target, d_best_candidate_index, d_min_diff);
            cudaMemcpy(&best_candidate_index, d_best_candidate_index, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_best_candidate_index);
            cudaFree(d_min_diff);

            replace_image_kernel<<<gridDim, blockDim>>>(*d_target_img, d_result_img, candidate_imgs[best_candidate_index], row, col);
        }
    }

    cudaMemcpy(result, d_result_img, sizeof(RGBImage), cudaMemcpyDeviceToHost);
    cudaFree(d_target_img);
    cudaFree(d_result_img);
    cudaFree(d_candidate_imgs);

    return result;
}
