#include <slime/SlimeExporter.hpp>
#include <glm/glm.hpp>
#include <stb_image_write.h>
#include <sstream>
#include <utils/CUDAUtils.h>

using namespace slime;



__global__ void Float2CharKernel(const float* dImage, unsigned char* dTex, unsigned int Width, unsigned int Height)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= Width || j >= Height)
        return;


    dTex[j * Width + i] = (unsigned char)glm::clamp(dImage[(Height - j - 1) * Width + i] * 255.0f, 0.0f, 255.0f);
}



void slime::SlimeExporter::ExportFrame(const float* dTexture)
{
    dim3 bSize(32, 32);
    dim3 gSize((Width + bSize.x - 1) / bSize.x, (Height + bSize.y - 1) / bSize.y);
    Float2CharKernel<<<gSize, bSize>>>(dTexture, dExpTex, Width, Height);
    cudaErrorCheck(cudaDeviceSynchronize());

    cudaErrorCheck(cudaMemcpy(ExpTex, dExpTex, Width * Height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    std::stringstream ss;
    ss << ExpDir << "/" << NumFrames++ << ".png";
    stbi_write_png(ss.str().c_str(), Width, Height, 1, ExpTex, Width);
}