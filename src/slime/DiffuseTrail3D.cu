#include <slime/SlimeSim3D.hpp>
#include <utils/CUDAUtils.h>



using namespace slime;



__global__ void DiffuseTrailKernel(float* TrailMap, float* DiffuseTrail, int Width, int Height, SimulationParameters Params)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Width || j >= Height)
        return;

    DiffuseTrail[j * Width + i] = TrailMap[j * Width + i];

    // Compute the mean with adjacent cells
    float Sum = 0.0f;
    for (int dx = -1; dx <= 1; ++dx)
    {
        int X = glm::clamp(i + dx, 0, Width - 1);
        for (int dy = -1; dy <= 1; ++dy)
        {
            int Y = glm::clamp(j + dy, 0, Height - 1);
            Sum += TrailMap[Y * Width + X];
        }
    }
    float Blur = Sum / 9.0f;
    float Orig = TrailMap[j * Width + i];
    float DiffuseWeight = glm::clamp(Params.DiffuseRate * Params.DeltaTime, 0.0f, 1.0f);
    float New = Orig * (1 - DiffuseWeight) + Blur * DiffuseWeight;
    DiffuseTrail[j * Width + i] = glm::max(0.0f, New - Params.DecayRate * Params.DeltaTime);
}



void slime::SlimeSim3D::LaunchDiffuseTrailKernel()
{
    int Width = TrailMapTex.Width;
    int Height = TrailMapTex.Height;
    dim3 bSize(32, 32);
    dim3 gSize((Width + bSize.x - 1) / bSize.x, (Height + bSize.y - 1) / bSize.y);
    DiffuseTrailKernel<<<gSize, bSize>>>(dTrailMap, dDiffuseTrail, Width, Height, Params);
    cudaErrorCheck(cudaDeviceSynchronize());
}