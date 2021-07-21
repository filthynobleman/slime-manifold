#include <slime/SlimeSim3D.hpp>
#include <utils/CUDAUtils.h>



using namespace slime;



__global__ void DiffuseTrailKernel(float* TrailMap, float* DiffuseTrail, int Width, int Height, SimulationParameters Params)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Width || j >= Height)
        return;

    for (int k = 0; k < Params.NumSpecies; ++k)
        DiffuseTrail[(j * Width + i) * Params.NumSpecies + k] = TrailMap[(j * Width + i) * Params.NumSpecies + k];

    // Compute the mean with adjacent cells
    for (int k = 0; k < Params.NumSpecies; ++k)
    {
        float Sum = 0.0f;
        for (int dx = -1; dx <= 1; ++dx)
        {
            int X = glm::clamp(i + dx, 0, Width - 1);
            for (int dy = -1; dy <= 1; ++dy)
            {
                int Y = glm::clamp(j + dy, 0, Height - 1);
                int Idx = Y * Width + X;
                Sum += TrailMap[Idx * Params.NumSpecies + k];
            }
        }
        float Blur = Sum / 9.0f;
        float Orig = TrailMap[(j * Width + i) * Params.NumSpecies + k];
        float DiffuseWeight = glm::clamp(Params.DiffuseRate * Params.DeltaTime, 0.0f, 1.0f);
        float New = Orig * (1 - DiffuseWeight) + Blur * DiffuseWeight;
        DiffuseTrail[(j * Width + i) * Params.NumSpecies + k] = glm::max(0.0f, New - Params.DecayRate * Params.DeltaTime);
    }
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