// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void castFloatToChar(float *inputImageData, unsigned char *ucharImage, int width, int height, int channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = width * height * channels;
  if (idx < size) {
      ucharImage[idx] = (unsigned char) (255 * inputImageData[idx]);
  }
}

__global__ void convertToGrayscale(unsigned char *ucharImage, unsigned char *grayImage, int width, int height) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < width * height) {
      unsigned char r = ucharImage[ii * 3];
      unsigned char g = ucharImage[ii * 3 + 1];
      unsigned char b = ucharImage[ii * 3 + 2];
      grayImage[ii] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}


__global__ void computeHistogram(unsigned char *grayImage, unsigned int *histogram, int width, int height) {
  __shared__ unsigned int private_hist[HISTOGRAM_LENGTH];
  if (threadIdx.x < 256) private_hist[threadIdx.x] = 0;
  __syncthreads();
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while (ii < width * height){
    atomicAdd( &(private_hist[grayImage[ii]]), 1);
    ii += stride;
  }
  __syncthreads();
  if (threadIdx.x < 256) {
      atomicAdd(&(histogram[threadIdx.x]), private_hist[threadIdx.x]);
  }
}

__global__ void Equalization(float *EqualizedCDF, float *cdf, float cdfmin){
  int i = threadIdx.x;
  EqualizedCDF[i] = (float) 255*(cdf[i] - cdfmin)/(1.0 - cdfmin);
  EqualizedCDF[i] = max(0.0f, EqualizedCDF[i]);
  EqualizedCDF[i] = min(255.0f, EqualizedCDF[i]);
}

__global__ void castToFloat(unsigned char *ucharImage, float *EqualizedCDF, float *output, int width, int height, int channels, float cdfmin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = width * height * channels;
  if (idx < size) {
      ucharImage[idx] = EqualizedCDF[ucharImage[idx]];
      output[idx] = (float) (ucharImage[idx]/255.0);
  }
}




int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char *deviceUcharImage;
  unsigned char *deviceGrayImage;
  unsigned int *deviceHistogram;
  float *deviceCDF;
  float *deviceEqualizedCDF;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int size = imageWidth * imageHeight;
  cudaMalloc((void **) &deviceInputImageData, imageChannels * size * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, imageChannels * size * sizeof(float));
  cudaMalloc((void **) &deviceUcharImage, imageChannels * size * sizeof(unsigned char));
  cudaMalloc((void **) &deviceGrayImage, size * sizeof(unsigned char));
  cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **) &deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **) &deviceEqualizedCDF, HISTOGRAM_LENGTH * sizeof(float));
  
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageChannels * size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));

  dim3 dimGrid(ceil(imageChannels * size / 256.0), 1, 1);
  dim3 dimBlock(256, 1, 1);
  castFloatToChar<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceUcharImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();


  dim3 dimGridGray(ceil(size / 256.0), 1, 1);
  convertToGrayscale<<<dimGridGray, dimBlock>>>(deviceUcharImage, deviceGrayImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dim3 dimGridHis(ceil(size / 512.0), 1, 1);
  computeHistogram<<<dimGridHis, 512>>>(deviceGrayImage, deviceHistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  unsigned int hostHistogram[HISTOGRAM_LENGTH];
  float hostCDF[HISTOGRAM_LENGTH];
  cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  hostCDF[0] = (float)hostHistogram[0] / size;
  for (int i=1; i < 256; i++){
    hostCDF[i] = hostCDF[i - 1] + (float)hostHistogram[i] / size;
  }
  float cdfmin = 0;
  for (int i=0; i < 256; i++){
    if (hostCDF[i] != 0){
      cdfmin = hostCDF[i];
      break;
    }
  }
  cudaMemcpy(deviceCDF, hostCDF, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

  Equalization<<<1,256>>>(deviceEqualizedCDF, deviceCDF, cdfmin);
  cudaDeviceSynchronize();

  castToFloat<<<dimGrid, dimBlock>>>(deviceUcharImage, deviceEqualizedCDF, deviceOutputImageData, imageWidth, imageHeight, imageChannels, cdfmin);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageChannels* size * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceUcharImage);
  cudaFree(deviceGrayImage);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}