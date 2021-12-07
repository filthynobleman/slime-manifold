#include <slime/SlimeSim3D.hpp>
#include <utils/CUDAUtils.h>
#include <glm/gtx/compatibility.hpp>


using namespace slime;



__global__ void InitAgentsKernel(Agent* Agents, mesh::Vertex* Verts, mesh::Triangle* Tris, float* TrailMap, unsigned char* StaticTrail,
                                 int Width, int Height, SimulationParameters Params)
{
    int AgentID = blockDim.x * blockIdx.x + threadIdx.x;
    if (AgentID >= Params.NumAgents)
        return;

    // Get the agent
    Agent A = Agents[AgentID];
    A.RandState = AgentID;
    A.RandState = RandHash(A.RandState);
    A.SpeciesID = A.RandState % Params.NumSpecies;
    
    // While agent position is inside an obstacle, recompute triangle and position
    bool InitOK = false;
    do
    {
        // Compute a random triangle
        A.RandState = RandHash(A.RandState);
        A.TriID = A.RandState % Params.NTris;
        
        // Compute random barycentric coordinates
        A.RandState = RandHash(A.RandState);
        float L1 = ScaleTo01(A.RandState);
        A.RandState = RandHash(A.RandState);
        float L2 = ScaleTo01(A.RandState);
        A.RandState = RandHash(A.RandState);
        float L3 = ScaleTo01(A.RandState);
        
        // Adjust barycentric coords
        float Sum = L1 + L2 + L3;
        assert(L1 >= 0 && L1 <= 1);    
        assert(L2 >= 0 && L2 <= 1);
        assert(L3 >= 0 && L3 <= 1);

        // Compute a random direction
        A.RandState = RandHash(A.RandState);
        A.Angle = ScaleTo01(A.RandState) * 3.14159265f;

        // Set the position using barycentric coords
        A.Pos = glm::vec2(0.0f, 0.0f);
        A.Pos += L1 * Verts[Tris[A.TriID].Verts[0]].TexUV;
        A.Pos += L2 * Verts[Tris[A.TriID].Verts[1]].TexUV;
        A.Pos += L3 * Verts[Tris[A.TriID].Verts[2]].TexUV;
        A.Pos /= Sum;
        // A.Pos = glm::clamp(A.Pos, glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));
        assert(A.Pos.y >= 0);
        assert(A.Pos.y <= 1);
        assert(A.Pos.x >= 0);
        assert(A.Pos.x <= 1);

        // Init the agent where needed
        if (StaticTrail != NULL)
        {
            int X = (int)(A.Pos.x * (Width - 1));
            int Y = (int)(A.Pos.y * (Height - 1));
            X = glm::clamp(X, 0, Width - 1);
            Y = glm::clamp(Y, 0, Height - 1);
            unsigned char InitVal = StaticTrail[3 * (Y * Width + X)];
            unsigned char ObstacleVal = StaticTrail[3 * (Y * Width + X) + 1];
            float InitProbability = (InitVal - ObstacleVal) / 255.0f;
            A.RandState = RandHash(A.RandState);
            if (ScaleTo01(A.RandState) <= InitProbability)
                InitOK = true;
        }
    }
    while (!InitOK);

    // Update the agent in the array
    Agents[AgentID] = A;
}




void slime::SlimeSim3D::LaunchInitAgentsKernel()
{
    dim3 bSize(1024);
    dim3 gSize((Params.NumAgents + bSize.x - 1) / bSize.x);
    InitAgentsKernel<<<gSize, bSize>>>(dAgents, dVerts, dTris, dTrailMap, dStaticTrail, TrailMapTex.Width, TrailMapTex.Height, Params);
    cudaErrorCheck(cudaDeviceSynchronize());
}