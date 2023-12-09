// #include <cmath>
// #include <iostream>
// #include <cuda_fp16.h>
// #include "gpu-new-forward.h"

// __constant__ float const_mask[4096];
// __global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input,const float * __restrict__ unused_mask,  const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;

//     int w = blockIdx.x * blockDim.x + threadIdx.x;
//     int h = blockIdx.y * blockDim.y + threadIdx.y;
//     int m = blockIdx.z;
//     if (w < W_out && h < H_out && m < M) {
//         for (int b = 0; b < B; b++) {
//             float accum = 0.0;
//             for (int c = 0; c < C; c += 2) { 
//                 for (int p = 0; p < K; p++) {
//                     for (int q = 0; q < K; q += 2) { 
//                         int h_in = h * S + p;
//                         int w_in = w * S + q;
//                         int input_index = b * (C * H * W) + c * (H * W) + h_in * W + w_in;
//                         int mask_index = m * (C * K * K) + c * (K * K) + p * K + q;
//                         accum += input[input_index] *  const_mask[mask_index];
//                         if (q + 1 < K) {
//                             int input_index2 = input_index + 1;
//                             int mask_index2 = mask_index + 1;
//                             accum += input[input_index2] *  const_mask[mask_index2];
//                         }
//                         if (c + 1 < C) {
//                             int input_index2 = input_index + (H * W);
//                             int mask_index2 = mask_index + (K * K);
//                             accum += input[input_index2] *  const_mask[mask_index2];
//                             if (q + 1 < K) {
//                                 int input_index3 = input_index2 + 1;
//                                 int mask_index3 = mask_index2 + 1;
//                                 accum += input[input_index3] *  const_mask[mask_index3];
//                             }
//                         }
//                     }
//                 }
//             }
//             output[b * (M * H_out * W_out) + m * (H_out * W_out) + h * W_out + w] = accum;
//         }
//     }
// }



// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     size_t input_size = B * C * H * W * sizeof(float);
//     size_t output_size = B * M * ((H - K) / S + 1) * ((W - K) / S + 1) * sizeof(float);
//     // size_t mask_size = M * C * K * K * sizeof(float);

//     cudaMalloc((void**)device_input_ptr, input_size);
//     cudaMalloc((void**)device_output_ptr, output_size);
//     cudaMalloc((void**)device_mask_ptr, 1);

//     cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
//     // cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
//     cudaMemset(*device_output_ptr, 0, output_size);
//     cudaMemcpyToSymbol(const_mask, host_mask, M * C * K * K * sizeof(float));
// }

// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
    
//     dim3 blockDim(8,8);
//     int gridW = (W_out + blockDim.x - 1) / blockDim.x;
//     int gridH = (H_out + blockDim.y - 1) / blockDim.y;

//     dim3 gridDim(gridW, gridH, M); 

//     conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask,B, M, C, H, W, K, S);
//     cudaDeviceSynchronize();
// }

// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;

//     size_t output_size = B * M * H_out * W_out * sizeof(float);
//     cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
   
//     cudaFree(device_input);
//     cudaFree(device_output);
//     cudaFree(device_mask);

// }