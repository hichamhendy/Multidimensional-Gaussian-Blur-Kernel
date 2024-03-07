# Multi-Dimensional Gaussian Blur for Video Processing

This repository contains code for performing multi-dimensional Gaussian blurring on video data. Unlike traditional blurring techniques that operate solely on RGB channels, this implementation considers the temporal aspect by processing multiple frames simultaneously.

## Overview

The core functionality of this code lies in applying Gaussian blur to each pixel of a video frame while considering neighboring pixels across two additional following frames. The process involves:

- Extracting 3D slices from the input video data.
- Applying a Gaussian blur filter to each slice.
- Aggregating the blurred pixel values to obtain the final output.

## Usage

To use the provided code, follow these steps:

1. **Setup Environment**: Ensure you have the necessary CUDA-compatible GPU and CUDA toolkit installed on your system.

2. **Compilation**: Compile the CUDA code using a compatible compiler. For example, you can use `nvcc`:

    ```bash
    make clean && make build
    ```

3. **Execution**: Execute the compiled binary, providing the necessary input parameters such as video dimensions and duration:

    ```bash
    ./blur_video <input_video_path> -t number_of_threads
    ```

## CUDA Kernels

### `multiDimensionalBlur` Kernel

This kernel operates on 3D data, processing each pixel across multiple frames to calculate Gaussian blur efficiently. It utilizes parallelism provided by CUDA threads to handle large datasets effectively.

### `\_3dSlice` Function

The `_3dSlice` function extracts a 3D slice from the input video data based on the specified dimensions. It efficiently maps thread indices to the corresponding elements in the 3D volume for processing.

### `\_3dGaussianBlurPixel` Function

This function computes the Gaussian blur for a single pixel by convolving the pixel's neighborhood with a predefined 3D Gaussian mask. It aggregates the weighted sum of neighboring pixel values to produce the blurred output.


### Additional Host Functions

- `allocateDeviceMemory`: Allocates GPU device memory for input and output video data.
- `copyFromHostToDevice`: Copies data from the host to the device, including input video data and the 3D Gaussian mask.
- `executeKernel`: Executes the CUDA kernel for Gaussian blur processing.
- `copyFromDeviceToHost`: Copies data from the device to the host, including the output video data.
- `deallocateMemory`: Frees GPU device memory after processing.
- `parseCommandLineArguments`: Parses command-line arguments for specifying input and output file paths, thread per block, and kernel dimension.
- `generate3DGaussian`: Generates a 3D Gaussian mask for Gaussian blur computation.
- `checkCudaCaps`: Checks CUDA capabilities of the GPU device for compatibility.


## Contributing

Contributions to this project are welcome. Feel free to open issues for bug reports or suggestions for improvement. Additionally, pull requests are encouraged for adding new features or optimizing existing code.

## License

This project is licensed under the [MIT License](LICENSE).
