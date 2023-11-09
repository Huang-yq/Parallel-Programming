// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define AUX_SIZE 1024

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, float *aux) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  __shared__ float temp[2 * BLOCK_SIZE];
  int tx = threadIdx.x;
  int start = 2 * blockIdx.x * blockDim.x;
  
  // Load data into shared memory
  if (start + tx < len)
    temp[tx] = input[start + tx];
  else
    temp[tx] = 0;
  if (start + tx + BLOCK_SIZE < len)
    temp[tx + BLOCK_SIZE] = input[start + tx + BLOCK_SIZE];
  else
    temp[tx + BLOCK_SIZE] = 0;
  
  // Reduce step
  for (int stride = 1; stride <2* BLOCK_SIZE; stride *= 2) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE && (index-stride)>=0)
      temp[index] += temp[index - stride];
  }
  
  // Post-reduction reverse step
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if ((index + stride) < 2 * BLOCK_SIZE)
      temp[index + stride] += temp[index];
  }
  
  // Write results to output and aux arrays
  __syncthreads();
  aux[blockIdx.x] = temp[2 * BLOCK_SIZE - 1];
  if (start + tx < len)
    output[start + tx] = temp[tx];
  if (start + tx + BLOCK_SIZE < len)
    output[start + tx + BLOCK_SIZE] = temp[tx + BLOCK_SIZE];
     
}

__global__ void addScannedBlockSums(float *output, float *aux, int len) {
  int tx = threadIdx.x;
  int start = 2 * blockDim.x * (blockIdx.x+1);
  
  if (start + tx < len)
    output[start + tx] += aux[blockIdx.x];
  if (start + tx + blockDim.x < len)
    output[start + tx + blockDim.x] += aux[blockIdx.x];
}





int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  float *deviceAux;
  float *auxSum;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  auxSum = (float *)malloc(AUX_SIZE * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAux, AUX_SIZE * sizeof(float)));
  cudaMemset(deviceAux, 0, AUX_SIZE * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(BLOCK_SIZE, 1, 1);
  dim3 gridDim(numElements / (2 * BLOCK_SIZE) + 1, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numElements, deviceAux);
  cudaDeviceSynchronize();
  //@@ Launch scan on aux array
  scan<<<1, blockDim>>>(deviceAux, deviceAux, AUX_SIZE, deviceInput);
  cudaDeviceSynchronize();
  //@@ Launch addScannedBlockSums kernel
  addScannedBlockSums<<<gridDim, blockDim>>>(deviceOutput, deviceAux, numElements);
  cudaDeviceSynchronize();
 
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAux);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  free(auxSum);
  return 0;
}
