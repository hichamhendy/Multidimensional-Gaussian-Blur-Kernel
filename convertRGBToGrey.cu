
#include "convertRGBToGrey.hpp"

/*
 * CUDA Kernel Device code
 *
 */
__global__ void multiDimensionalBlur(float* input3DData, float* output3DData)
{
    dim3 sliceDimensions(3, 3, 3);

    const int width = d_width;
    const int height = d_height;
    const int channels = d_channel;
    const int duration = d_duration;

    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int t   = threadIdx.z + blockIdx.z * blockDim.z;

    // Calculate the linear thread ID for the current channel
    const unsigned int threadId = (col + row * width + t * width * height) * channels;

    // Iterate over each channel
    for (int c = 0; c < channels; ++c)
    {
        if (row < height && col < width && t < duration)
        {
            float input3dDataSlice[27];
            const unsigned int channelIndex = threadId + c;

            // Extract the slice for the current channel
            _3dSlice(input3DData, sliceDimensions, input3dDataSlice, row, col, t, channelIndex);

            // Calculate blurred pixel value for the current channel's slice
            float blurredPixelValue = _3dGaussianBlurPixel(input3dDataSlice);

            // Store the blurred pixel value in the output array
            output3DData[channelIndex] = blurredPixelValue;

            // Perform any processing here, for now just copy the input to output
            // output3DData[channelIndex] = input3DData[channelIndex]; // Experiment succeeded
        }
    }
}


/**
 * @brief This function effectively extracts a 3D slice of data from the input 3D volume (input3dData) and stores it into the provided output slice (input3dDataSlice). 
 * Each thread handles copying a single element of the slice, ensuring parallelism across the 3D data volume.
 *  TODO: 
    * loop through z=[t_0, t_1, t_2]  
    * determine block id 
    * loop through x, y  +/- from current mapped x,y from block and thread index information  
    * determine thread id  
    * set x,y,z based on 0 with +/- from bottom left of t_0 
 * 
 * @param input3dData 
 * @param sliceDimensions 
 * @param input3dDataSlice 
 * @return void 
 */
__device__ void _3dSlice(float *input3dData, dim3 sliceDimensions, float *input3dDataSlice, int row, int column, int time, int threadId) 
{
    // Get the dimensions of the slice
    int sliceWidth = sliceDimensions.x;
    int sliceHeight = sliceDimensions.y;
    int sliceDepth = sliceDimensions.z;

    // Get the dimensions of the input 3D data
    int dataWidth = d_width;
    int dataHeight = d_height;
    int dataDepth = d_duration; 

    for(int k = 0; k <  sliceDepth/2; k++) 
    {
        for(int i = -sliceWidth/2; i<= sliceWidth/2; i++) 
        {
            for(int j = -sliceHeight/2; j <= sliceHeight/2; j++) 
            {
                const unsigned int x = max(0, min(dataWidth  - 1, column + j));
                const unsigned int y = max(0, min(dataHeight - 1, row    + i));
                const unsigned int z = max(0, min(dataDepth  - 1, time   + k));

                const unsigned int sliceIndex = x + y * dataWidth + z * (dataHeight * dataWidth);

                input3dDataSlice[j + i * sliceHeight + k * (sliceHeight * sliceWidth) ] = input3dData[threadId];
            }
        }
    }
}

__device__ float _3dGaussianBlurPixel(float * input3dDataSlice) 
{
    float pixelValueSum = 0.0f;

    //loop through x = [0,1,2] 
    //loop through y = [0,1,2]  
    //loop through z = [0,1,2] 

    // Loop through x, y, z = [0, 1, 2]
    for (int z = 0; z < 3; ++z)
    {
        for (int y = 0; y < 3; ++y)
        {
            for (int x = 0; x < 3; ++x)
            {
                // Calculate the 1D index for accessing the slice data
                int sliceIndex = x + y * 3 + z * (3 * 3);

                // Apply the mask and accumulate the result
                pixelValueSum += input3dDataSlice[sliceIndex] * d_3d_mask[x + y * 3 + z * (3 * 3)];
            }
        }
    }
    return pixelValueSum / d_mask_weight_sum; // Normalize by the sum of mask weights
}

__host__ std::tuple<float *, float *> allocateDeviceMemory(int width, int height, int duration, int channels)
{
    std::cout << "Allocating GPU device memory\n";
    int stream_size = width * height * duration * channels;
    size_t size = stream_size * sizeof(float);

    // Allocate the device input vector inputDeviceVideoData
    float *inputDeviceVideoData = NULL;
    cudaError_t err = cudaMalloc((void**) &inputDeviceVideoData, size);  // study: cudaMalloc((void**)& dIn, vBytes(hIn))
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector inputDeviceVideoData (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector inputDeviceVideoData
    float *outputDeviceVideoData = NULL;
    err = cudaMalloc((void**) &outputDeviceVideoData, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector outputDeviceVideoData; (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return {inputDeviceVideoData, outputDeviceVideoData};
}


__host__ void copyFromHostToDevice(float* h_input, float* d_input, float* h_mask_3d, int width, int height, int duration, int channels, int ker_dim)
{
    std::cout << "Copying from Host to Device\n";
    int stream_size = width * height * duration * channels;
    size_t size = stream_size * sizeof(float);

    cudaError_t err;
    err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector h_input from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int mask_weight_sum {0};
    for(int i = 0; i < ((int) pow(ker_dim, 3)); i++)
        mask_weight_sum += h_mask_3d[i];

    //Allocate device constant symbols for width and height and duration
    cudaMemcpyToSymbol(d_width, &width, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_height, &height, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_duration, &duration, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_channel, &channels, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask_weight_sum, &mask_weight_sum, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_3d_mask, h_mask_3d, ((int) pow(ker_dim,3)) * sizeof(float), 0, cudaMemcpyHostToDevice); //  h_mask_3d is already a pointer to the data, so it shouldn't be dereferenced again.

    cudaDeviceSynchronize();
}

__host__ void executeKernel(float *inputDeviceVideoData, float* outputDeviceVideoData, int threadsPerBlock, int width, int height,int duration, int channels)
{
    //Launch the convert CUDA Kernel
    std::cout << "Executing kernel\n";

    // Calculate the number of blocks needed in each dimension
    int blocks_x = (width * 1 + threadsPerBlock - 1) / threadsPerBlock;
    int blocks_y = (height * 1 + threadsPerBlock - 1) / threadsPerBlock;
    int blocks_z = (duration + threadsPerBlock - 1) / threadsPerBlock;

    // Define the block dimensions
    dim3 blockSize(threadsPerBlock, threadsPerBlock, threadsPerBlock);

    // Define the grid dimensions
    dim3 gridSize(blocks_x, blocks_y, blocks_z);

    printf("Calling the kernel for video of %d x %d x %d x %d\n", width, height, duration, channels);

    // Launch the kernel with the specified grid and block dimensions
    multiDimensionalBlur<<<gridSize, blockSize>>>(inputDeviceVideoData, outputDeviceVideoData);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void copyFromDeviceToHost(float *d_output, float* h_output, int width,int height,int duration, int channels)
{
    std::cout << "Copying from Device to Host\n";
    // Copy the device result int array in device memory to the host result int array in host memory.
    int stream_size = height * width * duration * channels;
    size_t size = stream_size * sizeof(float);

    cudaError_t err = cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array d_output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free device global memory
__host__ void deallocateMemory(float *d_input, float *d_output)
{
    std::cout << "Deallocating GPU device memory\n";
    cudaError_t err = cudaFree(d_input);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    std::cout << "Cleaning CUDA device\n";
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


__host__ void cleanUpHost(float* input_host_video_data, int width, int height, int duration)
{
    printf("Cleaning the host...\n\n");
/*     for (int i = 0; i < width; ++i) 
    {
        for (int j = 0; j < height; ++j) 
        {
            delete[] input_host_video_data[i][j];
        }
        delete[] input_host_video_data[i];
    } */
    delete[] input_host_video_data;
}

__host__ std::tuple<std::string, std::string, int, int> parseCommandLineArguments(int argc, char *argv[])
{
    std::cout << "Parsing CLI arguments\n";
    int threadsPerBlock = 256;
    int kernelDim = 3;
    std::string inputImage = "fast-x-teaser-trailer-2023-144.mp4";
    std::string outputImage = inputImage + "_gaussianBlurred.mp4";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-i") == 0)
        {
            inputImage = value;
            outputImage = inputImage + "_gaussianBlurred.mp4";
        }
        else if (option.compare("-o") == 0)
        {
            outputImage = value;
        }
        else if (option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if (option.compare("-k") == 0)
        {
            kernelDim = atoi(value.c_str());
        }
    }
    std::cout << "inputImage: " << inputImage << " outputImage: " << outputImage << " threadsPerBlock: " << threadsPerBlock 
        << " KernelDim: " << kernelDim << "\n";
    return {inputImage, outputImage, threadsPerBlock, kernelDim};
}


__host__ std::tuple<int, int, int, int, int, float*> readVideoFromFile(std::string inputFile) 
{
    cv::VideoCapture cap(inputFile);

    if (!cap.isOpened()) {
        std::cerr << "Error: Couldn't open the video file.\n";
        exit(EXIT_FAILURE);
    }

    // Check if the input video file contains playable streams
    if (cap.get(cv::CAP_PROP_FRAME_COUNT) == 0) {
        std::cerr << "Error: Input video file contains no playable streams.\n";
        exit(EXIT_FAILURE);
    }
    
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int channels = 3; // cap.get(cv::CAP_PROP_CHANNEL); // Video input or Channel Number (only for those cameras that support)  Assuming RGB color space
    int duration = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    
    float *output_host_video_data = (float *)malloc(sizeof(float) * width * height * channels * duration);

    std::cout << "Video width: " << width << " height: " << height << " duration: " << duration  << " fbs: " << fps << " channel: " << channels << std::endl;

    float *inputHostVideoData = (float *)malloc(sizeof(float) * width * height * channels * duration);

    std::cout << "Memory allocation on host is successfully done" << std::endl;

    cv::Mat frame;
    for (int t = 0; t < duration; ++t) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed.\n";
            cleanUpHost(inputHostVideoData, width, height, duration);
            exit(EXIT_FAILURE);
        }
        
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                for (int c = 0; c < channels; ++c) {
                    if (frame.empty()) {
                        std::cerr << "Error: Video file does not contain enough frames.\n";
                        std::cout << "stopped at " << i << " " << j << " " << t << std::endl;
                        cleanUpHost(inputHostVideoData, width, height, duration);
                        exit(EXIT_FAILURE);
                    }
                    inputHostVideoData[t * width * height * channels + j * width * channels + i * channels + c] = static_cast<float>(frame.at<cv::Vec3b>(j, i)[c]) / 255.0;  //  at (int row, int col)
                    
                    output_host_video_data[t * width * height * channels + j * width * channels + i * channels + c] = inputHostVideoData[t * width * height * channels + j * width * channels + i * channels + c];
                }
            }
        }
    }

    cap.release();

    return {width, height, duration, channels, fps, inputHostVideoData};
}


__host__ void storeVideoData(float* outputHostVideoData, int width, int height, int duration, std::string outputFile) 
{
    std::cout << "Start storing the result" << std::endl;

    cv::VideoWriter videoWriter;
    int channels = 3; // constant assumption

    // Define the codec and create VideoWriter object
    int codec = cv::VideoWriter::fourcc('H', '2', '6', '4'); // H.264 codec
    double fps = 30.0; // You can adjust the frames per second as needed
    cv::Size frameSize(width, height);

    videoWriter.open(outputFile, codec, fps, frameSize, true);

    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Couldn't open the video writer.\n";
        exit(EXIT_FAILURE);
    }

    cv::Mat frame(height, width, CV_8UC3); // CV_8UC3 for 3-channel image (e.g., RGB)

    for (int t = 0; t < duration; ++t) 
    {
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                // Iterate over each channel
                for (int c = 0; c < channels; ++c) {
                    float pixelValue = outputHostVideoData[t * width * height * channels + (j * width + i) * channels + c];
                    // Set the pixel value in the frame
                    frame.at<cv::Vec3b>(j, i)[c] = static_cast<uchar>(pixelValue * 255.0);
                }
            }
        } 

        if (frame.empty()) {
            std::cerr << "Error: Video file does not contain enough frames.\n";
            exit(EXIT_FAILURE);
        }
        videoWriter.write(frame);

        // Display the resulting frame    
        cv::imshow("Results Frame", frame);
        // Press  ESC on keyboard to  exit
        char c = (char) cv::waitKey(10);
        if( c == 27 ) 
            break;
    }

    std::cout << "Storing ended. Check results!" << std::endl;
    cv::destroyAllWindows();
    videoWriter.release();
}


__host__ void generate3DGaussian(float* kernel, int dim, int radius) 
{
    float stdev = 1.0;
    float pi = 3.14159265358979323846;
    float constant = 10.0 / (2.0 * pi * stdev * stdev);

    
        for (int k = 0; k < dim; ++k) {
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    float exponent = -((i - radius) * (i - radius) + (j - radius) * (j - radius) + (k - radius) * (k - radius)) / (2 * stdev * stdev);
                    kernel[k * dim * dim + i * dim + j] = constant * exp(exponent);
                }
            }
        } 
}



__host__ bool checkCudaCaps()
{
    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if(deviceCount == 0)
    {
        std::cerr << "No CUDA-capable device found" << std::endl;
        return false;
    }
    std::cout << "CUDA-capable devices detected: " << deviceCount << std::endl;

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
            (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
            (runtimeVersion % 100) / 10);

    for(int device= 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " <<  deviceProp.name << std::endl;
        std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl; 
        std::cout << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        int maxThreads = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount;
        int maxBlocks = maxThreads / deviceProp.maxThreadsPerBlock;
        std::cout << "Max Blocks: " << maxBlocks << std::endl;
    }
  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}


int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    if (checkCudaCaps() == false)
    {
      exit(EXIT_SUCCESS);
    }

    std::tuple<std::string, std::string, int, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
    std::string input_video = std::get<0>(parsedCommandLineArgsTuple);
    std::string output_video = std::get<1>(parsedCommandLineArgsTuple);
    const int threads_per_block = std::get<2>(parsedCommandLineArgsTuple);
    const int conv_k_dim = std::get<3>(parsedCommandLineArgsTuple);
    try 
    {
        printf("Starting Reading...\n\n");
        std::tuple<int, int, int, int, int, float*> readTuple = readVideoFromFile(input_video);
        int width = std::get<0>(readTuple);
        int height = std::get<1>(readTuple);
        int duration = std::get<2>(readTuple);
        int channels = std::get<3>(readTuple);
        int fps = std::get<4>(readTuple);
        float* input_host_video_data = std::get<5>(readTuple);
        // float output_host_video_data[width][height][duration];
        printf("Reading the video done...\n\n");
        if ((height < 2 * conv_k_dim + 1) || (width < 2 * conv_k_dim + 1)) 
        {
            std::cout << "Image is too small to apply kernel effectively." << std::endl;
            exit(EXIT_FAILURE);
	    }
        float mask_3d[conv_k_dim * conv_k_dim * conv_k_dim];
        int k_radius = floor( conv_k_dim/ 2.0);
        generate3DGaussian(mask_3d, conv_k_dim, k_radius);
        std::cout << "The mask elments are ";
        for(int i = 0; i < ((int) pow(conv_k_dim, 3)); i++)
            std::cout << mask_3d[i] << " ";
        std::cout << std::endl;


        float *output_host_video_data = (float *)malloc(sizeof(float) * width * height * duration * channels);

        std::tuple<float *, float *> memoryTuple = allocateDeviceMemory(width, height, duration, channels);
        float *input_device_video_data = std::get<0>(memoryTuple);
        float *output_device_video_data = std::get<1>(memoryTuple);

        copyFromHostToDevice(input_host_video_data, input_device_video_data, mask_3d, width, height, duration, channels,conv_k_dim);

        executeKernel(input_device_video_data, output_device_video_data, threads_per_block, width, height, duration, channels);

        copyFromDeviceToHost(output_device_video_data, output_host_video_data, width, height, duration, channels);
        deallocateMemory(input_device_video_data, output_device_video_data);

        storeVideoData(output_host_video_data , width, height, duration,  output_video);
        
        cleanUpDevice();
        cleanUpHost(input_host_video_data, width, height, duration);
        delete output_host_video_data;
    }
    catch (std::exception &error_)
    {
        std::cout << "Caught exception: " << error_.what() << std::endl;
        return 1;
    }
    return 0;
}