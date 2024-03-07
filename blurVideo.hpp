#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <helper_cuda.h>

#include <iostream>
#include <stdio.h>
#include <tuple>
#include <string>
#include <stdexcept>
#include <iostream>


__device__ __constant__ int d_width;
__device__ __constant__ int d_height;
__device__ __constant__ int d_duration;
__device__ __constant__ int d_channel;
__device__ __constant__ int d_mask_weight_sum;// = 38;
__device__ __constant__ float d_3d_mask[27];// = {1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 4, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1};



__device__ float _3dGaussianBlurPixel(float * input3dDataSlice);
__device__ void _3dSlice(float *input3dData, dim3 sliceDimensions, float *input3dDataSlice, int row, int column, int time, int threadId);
__global__ void multiDimensionalBlur(float* input3DData, float* output3DData);
__host__ std::tuple<float *, float *> allocateDeviceMemory(int width, int height, int duration, int channels);
__host__ void copyFromHostToDevice(float* h_input, float* d_input, float* h_mask_3d, int width, int height, int duration, int ker_dim = 3);
__host__ void executeKernel(float *inputDeviceVideoData, float* outputDeviceVideoData, int threadsPerBlock, int width,int height,int duration, int channels);
__host__ void copyFromDeviceToHost(float *d_output, float* h_output, int width,int height,int duration, int channels);
__host__ void deallocateMemory(float *d_input, float *d_output);
__host__ void cleanUpDevice();
__host__ void cleanUpHost(float* input_host_video_data, int width,int height,int duration);

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return std::tuple<std::string, std::string, int, int> 
 */
__host__ std::tuple<std::string, std::string, int, int> parseCommandLineArguments(int argc, char *argv[]);
__host__ std::tuple<int, int, int, int, int, float*> readVideoFromFile(std::string inputFile);
__host__ void storeVideoData(float* outputHostVideoData ,int width,int height,int duration, std::string outputFile);

/**
 * @brief The values from this function will create the convolution matrix / kernel that we’ll apply to every pixel in the original image. 
 * The kernel is typically quite small — the larger it is the more computation we have to do at every pixel.
 * x and y specify the delta from the center pixel (0, 0). For example, if the selected radius for the kernel was 3, x and y would range from -3 to 3 (inclusive).
 * σ – the standard deviation – influences how significantly the center pixel’s neighboring pixels affect the computations result.
 * 
 * @param kernel 
 * @param dim 
 * @param radius 
 * @return void 
 */
__host__ void generate3DGaussian(float* kernel, int dim, int radius);

__host__ bool checkCudaCaps();
