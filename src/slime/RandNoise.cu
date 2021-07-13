#include <utils/CUDAUtils.h>


__global__ void RandomNoiseKernel(float* Image, int Width, int Height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Width || j >= Height)
        return;

    int Idx = j * Width + i;
    if (Idx % 2 == 0)
        Image[Idx] = 0.0f;
    else
        Image[Idx] = 1.0f;
}



void RandomNoise(float *dImage, int Width, int Height)
{
    // dim3 bSize(1024);
    // dim3 gSize((Width * Height + bSize.x - 1) / bSize.x);
    dim3 bSize(32, 32);
    dim3 gSize((Width + bSize.x - 1) / bSize.x, (Height + bSize.y - 1) / bSize.y);

    // cudaErrorCheck(cudaBindSurfaceToArray(cudaSurf, dImage));
    // cudaErrorCheck(cudaBindTextureToArray(cudaTex, dImage));
    
    std::cout << bSize.x << " x " << bSize.y << std::endl;
    std::cout << gSize.x << " x " << gSize.y << std::endl;
    RandomNoiseKernel<<<gSize,bSize>>>(dImage, Width, Height);
    cudaErrorCheck(cudaDeviceSynchronize());
}