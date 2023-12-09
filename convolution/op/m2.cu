// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"
// void checkCudaErrors(cudaError_t err) {
//     if (err != cudaSuccess) {
//         std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }


// Just a copy for the original milestone 2

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// {

//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;

//     int w = blockIdx.x * blockDim.x + threadIdx.x;
//     int h = blockIdx.y * blockDim.y + threadIdx.y;
//     int m = blockIdx.z; 

//     if (w < W_out && h < H_out && m < M) {
//         for (int b = 0; b < B; b++) {
//             float accum = 0.0;
//             for (int c = 0; c < C; c++) { 
//                 for (int p = 0; p < K; p++) { 
//                     for (int q = 0; q < K; q++) {
//                         int h_in = h * S + p;
//                         int w_in = w * S + q;
//                         accum += input[b * (C * H * W) + c * (H * W) + h_in * W + w_in] * mask[m * (C * K * K) + c * (K * K) + p * K + q];
//                     }
//                 }
//             }
//             output[b * (M * H_out * W_out) + m * (H_out * W_out) + h * W_out + w] = accum;
//         }
//     }
// // }
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     // Allocate memory and copy over the relevant data structures to the GPU

//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     cudaError_t err;
//     size_t input_size = B * C * H * W * sizeof(float);
//     size_t output_size = B * M * ((H - K) / S + 1) * ((W - K) / S + 1) * sizeof(float);
//     size_t mask_size = M * C * K * K * sizeof(float);

//     err = cudaMalloc((void**)device_input_ptr, input_size);
//     checkCudaErrors(err);
//     err = cudaMalloc((void**)device_output_ptr, output_size);
//     checkCudaErrors(err);
//     err = cudaMalloc((void**)device_mask_ptr, mask_size);
//     checkCudaErrors(err);

//     err = cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
//     checkCudaErrors(err);
//     err = cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
//     checkCudaErrors(err);
//     err = cudaMemset(*device_output_ptr, 0, output_size);
//     checkCudaErrors(err);
// }

// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
    
//     dim3 blockDim(16, 16);
//     int gridW = (W_out + blockDim.x - 1) / blockDim.x;
//     int gridH = (H_out + blockDim.y - 1) / blockDim.y;

//     dim3 gridDim(gridW, gridH, M); 
//     conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    
//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaDeviceSynchronize());

// }



// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     // Copy the output back to host
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;

//     size_t output_size = B * M * H_out * W_out * sizeof(float);
//     checkCudaErrors(cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost));
   
//     checkCudaErrors(cudaFree(device_input));
//     checkCudaErrors(cudaFree(device_output));
//     checkCudaErrors(cudaFree(device_mask));

// }