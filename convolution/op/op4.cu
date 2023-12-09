// #include <cmath>
// #include <iostream>
// #include <cuda_fp16.h>
// #include "gpu-new-forward.h"

// #define TILE_WIDTH 16

// // Tuning with restrict and loop unrolling 
// // Different parameters for manual unrolling is commented out
// __global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;

//     int w = blockIdx.x * blockDim.x + threadIdx.x;
//     int h = blockIdx.y * blockDim.y + threadIdx.y;
//     int m = blockIdx.z; 

//     if (w < W_out && h < H_out && m < M) {
//         for (int b = 0; b < B; b++) {
//             float accum = 0.0;
            
//                 if (K<=7){
                    
//                     for (int c = 0; c < C; c++) {
//                         #pragma unroll
//                         for (int p = 0; p < K; p++) {
                            
//                             for (int q = 0; q < K; q++) {
//                                 int h_in = h * S + p;
//                                 int w_in = w * S + q;
//                                 int input_index = b * (C * H * W) + c * (H * W) + h_in * W + w_in;
//                                 int mask_index = m * (C * K * K) + c * (K * K) + p * K + q;
//                                 accum += input[input_index] * mask[mask_index];
//                             }
//                         }
//                     }
//                 }
//                 else{
//                     for (int p = 0; p < K; p++) {
//                         for (int q = 0; q < K; q++) {
//                             #pragma unroll 
//                             for (int c = 0; c < C; c++) {
//                                 int h_in = h * S + p;
//                                 int w_in = w * S + q;
//                                 int input_index = b * (C * H * W) + c * (H * W) + h_in * W + w_in;
//                                 int mask_index = m * (C * K * K) + c * (K * K) + p * K + q;
//                                 accum += input[input_index] * mask[mask_index];
//                             }
//                         }   
//                     }
//                 }
                
            
//             output[b * (M * H_out * W_out) + m * (H_out * W_out) + h * W_out + w] = accum;
//         }
//     }
// }

// // __global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// // {
// //     const int H_out = (H - K) / S + 1;
// //     const int W_out = (W - K) / S + 1;

// //     int w = blockIdx.x * blockDim.x + threadIdx.x;
// //     int h = blockIdx.y * blockDim.y + threadIdx.y;
// //     int m = blockIdx.z; 

// //     if (w < W_out && h < H_out && m < M) {
// //         for (int b = 0; b < B; b++) {
// //             float accum = 0.0;
// //             for (int c = 0; c < C; c += 2) { // Unroll by a factor of 2 over C
// //                 for (int p = 0; p < K; p++) {
// //                     for (int q = 0; q < K; q++) {
// //                         int h_in = h * S + p;
// //                         int w_in = w * S + q;
// //                         int input_index = b * (C * H * W) + c * (H * W) + h_in * W + w_in;
// //                         int mask_index = m * (C * K * K) + c * (K * K) + p * K + q;
// //                         accum += input[input_index] * mask[mask_index];

// //                         // Unroll the second iteration over C
// //                         if (c + 1 < C) {
// //                             int input_index2 = input_index + (H * W);
// //                             int mask_index2 = mask_index + (K * K);
// //                             accum += input[input_index2] * mask[mask_index2];
// //                         }
// //                     }
// //                 }
// //             }
// //             output[b * (M * H_out * W_out) + m * (H_out * W_out) + h * W_out + w] = accum;
// //         }
// //     }
// // }

// // __global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// // {
// //     const int H_out = (H - K) / S + 1;
// //     const int W_out = (W - K) / S + 1;

// //     int w = blockIdx.x * blockDim.x + threadIdx.x;
// //     int h = blockIdx.y * blockDim.y + threadIdx.y;
// //     int m = blockIdx.z;

// //     if (w < W_out && h < H_out && m < M) {
// //         for (int b = 0; b < B; b++) {
// //             float accum = 0.0;
// //             for (int c = 0; c < C; c ++) { // Unroll by a factor of 2 over C
// //                 for (int p = 0; p < K; p++) {
// //                     for (int q = 0; q < K; q += 3) { // Unroll by a factor of 3 over K
// //                         int h_in = h * S + p;
// //                         int w_in = w * S + q;
// //                         int input_index = b * (C * H * W) + c * (H * W) + h_in * W + w_in;
// //                         int mask_index = m * (C * K * K) + c * (K * K) + p * K + q;
// //                         accum += input[input_index] * mask[mask_index];

// //                         // Check if q+1 < K before unrolling the second iteration over K
// //                         if (q + 1 < K) {
// //                             int w_in2 = w * S + q + 1;
// //                             int input_index2 = b * (C * H * W) + c * (H * W) + h_in * W + w_in2;
// //                             int mask_index2 = mask_index + 1;
// //                             accum += input[input_index2] * mask[mask_index2];
// //                         }

// //                         // Check if q+2 < K before unrolling the third iteration over K
// //                         if (q + 2 < K) {
// //                             int w_in3 = w * S + q + 2;
// //                             int input_index3 = b * (C * H * W) + c * (H * W) + h_in * W + w_in3;
// //                             int mask_index3 = mask_index + 2;
// //                             accum += input[input_index3] * mask[mask_index3];
// //                         }
// //                     }
// //                 }
// //             }
// //             output[b * (M * H_out * W_out) + m * (H_out * W_out) + h * W_out + w] = accum;
// //         }
// //     }
// // }

// // __global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// // {
// //     const int H_out = (H - K) / S + 1;
// //     const int W_out = (W - K) / S + 1;

// //     int w = blockIdx.x * blockDim.x + threadIdx.x;
// //     int h = blockIdx.y * blockDim.y + threadIdx.y;
// //     int m = blockIdx.z;

// //     if (w < W_out && h < H_out && m < M) {
// //         for (int b = 0; b < B; b++) {
// //             float accum = 0.0;
// //             for (int c = 0; c < C; c += 2) { // Unroll by a factor of 2 over C
// //                 for (int p = 0; p < K; p += 2) { // Unroll by a factor of 2 over P
// //                     for (int q = 0; q < K; q += 2) { // Unroll by a factor of 2 over Q
// //                         int h_in = h * S + p;
// //                         int w_in = w * S + q;
// //                         int input_index = b * (C * H * W) + c * (H * W) + h_in * W + w_in;
// //                         int mask_index = m * (C * K * K) + c * (K * K) + p * K + q;
// //                         accum += input[input_index] * mask[mask_index];

// //                         // Additional bounds checks and unrolled computations
// //                         // for q+1, p+1, and c+1
// //                         if (q + 1 < K) {
// //                             accum += input[input_index + 1] * mask[mask_index + 1];
// //                         }
// //                         if (c + 1 < C) {
// //                             int input_index_c = input_index + (H * W);
// //                             int mask_index_c = mask_index + (K * K);
// //                             accum += input[input_index_c] * mask[mask_index_c];

// //                             if (q + 1 < K) {
// //                                 accum += input[input_index_c + 1] * mask[mask_index_c + 1];
// //                             }
// //                         }
// //                         if (p + 1 < K) {
// //                             int h_in_p = h_in + 1;
// //                             int input_index_p = b * (C * H * W) + c * (H * W) + h_in_p * W + w_in;
// //                             int mask_index_p = m * (C * K * K) + c * (K * K) + (p + 1) * K + q;
// //                             accum += input[input_index_p] * mask[mask_index_p];

// //                             if (q + 1 < K) {
// //                                 accum += input[input_index_p + 1] * mask[mask_index_p + 1];
// //                             }
// //                             if (c + 1 < C) {
// //                                 int input_index_cp = input_index_p + (H * W);
// //                                 int mask_index_cp = mask_index_p + (K * K);
// //                                 accum += input[input_index_cp] * mask[mask_index_cp];

// //                                 if (q + 1 < K) {
// //                                     accum += input[input_index_cp + 1] * mask[mask_index_cp + 1];
// //                                 }
// //                             }
// //                         }
// //                     }
// //                 }
// //             }
// //             output[b * (M * H_out * W_out) + m * (H_out * W_out) + h * W_out + w] = accum;
// //         }
// //     }
// // }
// // __global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// // {
// //     const int H_out = (H - K) / S + 1;
// //     const int W_out = (W - K) / S + 1;

// //     int w = blockIdx.x * blockDim.x + threadIdx.x;
// //     int h = blockIdx.y * blockDim.y + threadIdx.y;
// //     int m = blockIdx.z;

// //     if (w < W_out && h < H_out && m < M) {
// //         for (int b = 0; b < B; b++) {
// //             float accum = 0.0;
// //             for (int c = 0; c < C; c++) { // No unrolling over C
// //                 for (int p = 0; p < K; p += 2) { // Unroll by a factor of 2 over P
// //                     for (int q = 0; q < K; q += 2) { // Unroll by a factor of 2 over Q
// //                         int h_in = h * S + p;
// //                         int w_in = w * S + q;
// //                         int input_index = b * (C * H * W) + c * (H * W) + h_in * W + w_in;
// //                         int mask_index = m * (C * K * K) + c * (K * K) + p * K + q;

// //                         // Main accumulation
// //                         accum += input[input_index] * mask[mask_index];

// //                         // Unroll the loop for q
// //                         if (q + 1 < K) {
// //                             accum += input[input_index + 1] * mask[mask_index + 1];
// //                         }

// //                         // Unroll the loop for p
// //                         if (p + 1 < K) {
// //                             int h_in_p = h_in + 1;
// //                             int input_index_p = b * (C * H * W) + c * (H * W) + h_in_p * W + w_in;
// //                             int mask_index_p = m * (C * K * K) + c * (K * K) + (p + 1) * K + q;
// //                             accum += input[input_index_p] * mask[mask_index_p];

// //                             if (q + 1 < K) {
// //                                 accum += input[input_index_p + 1] * mask[mask_index_p + 1];
// //                             }
// //                         }
// //                     }
// //                 }
// //             }
// //             output[b * (M * H_out * W_out) + m * (H_out * W_out) + h * W_out + w] = accum;
// //         }
// //     }
// // }




// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     size_t input_size = B * C * H * W * sizeof(float);
//     size_t output_size = B * M * ((H - K) / S + 1) * ((W - K) / S + 1) * sizeof(float);
//     size_t mask_size = M * C * K * K * sizeof(float);

//     cudaMalloc((void**)device_input_ptr, input_size);
//     cudaMalloc((void**)device_output_ptr, output_size);
//     cudaMalloc((void**)device_mask_ptr, mask_size);

//     cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
//     cudaMemset(*device_output_ptr, 0, output_size);
// }

// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
// {
//     const int H_out = (H - K) / S + 1;
//     const int W_out = (W - K) / S + 1;
    
//     dim3 blockDim(8,8);
//     int gridW = (W_out + blockDim.x - 1) / blockDim.x;
//     int gridH = (H_out + blockDim.y - 1) / blockDim.y;

//     dim3 gridDim(gridW, gridH, M); 

//     conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
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



