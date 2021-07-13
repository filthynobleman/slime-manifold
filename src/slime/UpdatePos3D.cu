#include <slime/SlimeSim3D.hpp>
#include <utils/CUDAUtils.h>



using namespace slime;


struct EndPos
{
    glm::vec2   Pos;
    int         TriID;
};

__device__ glm::vec3 D2ToBaryc(glm::vec2 Coords, glm::vec2 V1, glm::vec2 V2, glm::vec2 V3)
{
    float T11 = V1.x - V3.x; // x1 - x3
    float T12 = V2.x - V3.x; // x2 - x3
    float T21 = V1.y - V3.y; // y1 - y3
    float T22 = V2.y - V3.y; // y2 - y3
    float DetT = T11 * T22 - T12 * T21;

    // Lambda1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / Det(T)
    float L1 = (T22 * (Coords.x - V3.x) - T12 * (Coords.y - V3.y)) / DetT;
    // Lambda2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / Det(T)
    float L2 = (T11 * (Coords.y - V3.y) - T21 * (Coords.x - V3.x)) / DetT;
    // Lambda3 = 1 - Lambda1 - Lambda2
    return glm::vec3(L1, L2, 1 - L1 - L2);
}

__device__ glm::vec3 D3ToBaryc(glm::vec3 Coords, glm::vec3 P1, glm::vec3 P2, glm::vec3 P3)
{
    glm::vec3 L(0.0f, 0.0f, 0.0f);

    // Compute voronoi areas
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float s = 0.0f;
    // P1 - P2 - C
    a = glm::length(P1 - P2);
    b = glm::length(P2 - Coords);
    c = glm::length(Coords - P1);
    s = (a + b + c) / 2.0f;
    L.z = glm::sqrt(s * (s - a) * (s - b) * (s - c));
    // P3 - P1 - C
    a = glm::length(P3 - P1);
    b = glm::length(P1 - Coords);
    c = glm::length(Coords - P3);
    s = (a + b + c) / 2.0f;
    L.y = glm::sqrt(s * (s - a) * (s - b) * (s - c));

    // Compute triangle area
    a = glm::length(P1 - P2);
    b = glm::length(P2 - P3);
    c = glm::length(P3 - P1);
    s = (a + b + c) / 2.0f;
    s = glm::sqrt(s * (s - a) * (s - b) * (s - c));
    
    // Return
    L /= s;
    L.x = 1 - L.z - L.y;
    return L;
}

__device__ glm::vec2 BarycToD2(glm::vec3 Coords, glm::vec2 V1, glm::vec2 V2, glm::vec2 V3)
{
    return Coords.x * V1 + Coords.y * V2 + Coords.z * V3;
}

__device__ glm::vec3 BarycToD3(glm::vec3 Coords, glm::vec3 V1, glm::vec3 V2, glm::vec3 V3)
{
    return Coords.x * V1 + Coords.y * V2 + Coords.z * V3;
}

__device__ glm::vec4 ProjOnTri(glm::vec3 P, glm::vec3 P1, glm::vec3 P2, glm::vec3 P3)
{
    // Get edges
    glm::vec3 E12 = P2 - P1;
    glm::vec3 E23 = P3 - P2;
    glm::vec3 E31 = P1 - P3;

    // Compute normal
    glm::vec3 N = glm::cross(E12, -E31);
    N = N + glm::cross(E23, -E12);
    N = N + glm::cross(E31, -E23);
    N = glm::normalize(N);

    // Compute centre
    glm::vec3 C = (P1 + P2 + P3) / 3.0f;

    // Project CP onto normal
    float Proj = glm::dot(P - C, N);

    // Remove projection from P
    return glm::vec4(P - Proj * N, Proj);
}


__device__ EndPos CalcEndPosition(glm::vec2 Pos, int TriID, mesh::Triangle* Tris, mesh::Vertex* Verts, glm::ivec3* T2T)
{
    EndPos ep;
    ep.Pos = Pos;
    ep.TriID = TriID;

    // Get the vertex indices of the triangles
    glm::ivec3 TV = Tris[TriID].Verts;

    // Get the vertices
    glm::vec2 UV1 = Verts[TV[0]].TexUV;
    glm::vec2 UV2 = Verts[TV[1]].TexUV;
    glm::vec2 UV3 = Verts[TV[2]].TexUV;

    // Convert to barycentric
    glm::vec3 L = D2ToBaryc(ep.Pos, UV1, UV2, UV3);

    // Non-negative coordinates = point inside
    if (L.x >= 0 && L.y >= 0 && L.z >= 0)
        return ep;  // Input position was good

    // Get vertices positions
    glm::vec3 V1 = Verts[TV[0]].Position;
    glm::vec3 V2 = Verts[TV[1]].Position;
    glm::vec3 V3 = Verts[TV[2]].Position;

    // Go to 3D space
    glm::vec3 P3D = BarycToD3(L, V1, V2, V3);

    // Search the neirest adjacent triangle
    glm::vec4 ProjDist(0.0f, 0.0f, 0.0f, 1e9f);
    int Neirest = 0;
    int i = 0;
    for (i = 0; i < 3; ++i)
    {
        // Get adjacent triangle and its vertices
        int AdjTri = T2T[TriID][i];
        glm::vec3 AV1 = Verts[Tris[AdjTri].Verts[0]].Position;
        glm::vec3 AV2 = Verts[Tris[AdjTri].Verts[1]].Position;
        glm::vec3 AV3 = Verts[Tris[AdjTri].Verts[2]].Position;
        // Project
        glm::vec4 ProjDistLoc = ProjOnTri(P3D, AV1, AV2, AV3);
        // If distance is (in absolute) less than previous, set current triangle
        if (abs(ProjDistLoc.w) < abs(ProjDist.w))
        {
            ProjDist = ProjDistLoc;
            Neirest = AdjTri;
        }
    }

    // Convert to barycentric in adjacent triangle
    TV = Tris[Neirest].Verts;
    UV1 = Verts[TV[0]].TexUV;
    UV2 = Verts[TV[1]].TexUV;
    UV3 = Verts[TV[2]].TexUV;
    V1 = Verts[TV[0]].Position;
    V2 = Verts[TV[1]].Position;
    V3 = Verts[TV[2]].Position;
    glm::vec3 Proj(ProjDist.x, ProjDist.y, ProjDist.z);
    L = D3ToBaryc(Proj, V1, V2, V3);
    // Adjust barycentric
    L /= (L.x + L.y + L.z);
    // And from barycentric to texture
    ep.Pos = BarycToD2(L, UV1, UV2, UV3);

    // Update triangle id and return
    ep.TriID = Neirest;
    return ep;
}

__device__ float Sense(glm::vec2 Centre, float* TrailMap, int Width, int Height, SimulationParameters Params)
{
    glm::ivec2 Coords(int(Centre.x * (Width - 1)), int(Centre.y * (Height - 1)));
    float Sum = 0.0f;
    for (int dx = -Params.SensorRadius; dx <= Params.SensorRadius; ++dx)
    {
        int X = glm::clamp(Coords.x + dx, 0, Width - 1);
        for (int dy = -Params.SensorRadius; dy <= Params.SensorRadius; ++dy)
        {
            int Y = glm::clamp(Coords.y + dy, 0, Height - 1);
            Sum += TrailMap[Y * Width + X];
        }
    }
    return Sum;
}

__device__ Agent NextDir(Agent A, float* TrailMap, int Width, int Height, 
                         mesh::Vertex* Verts, mesh::Triangle* Tris, glm::ivec3* T2T, 
                         SimulationParameters Params)
{
    // Get direction
    glm::vec2 Dir(glm::cos(A.Angle), glm::sin(A.Angle));

    // Look forward
    EndPos SensorFwd = CalcEndPosition(A.Pos + Dir * Params.VisionDist, A.TriID, Tris, Verts, T2T);
    float Fwd = Sense(SensorFwd.Pos, TrailMap, Width, Height, Params);

    // Look right
    Dir.x = glm::cos(A.Angle + glm::radians(Params.VisionAngle));
    Dir.y = glm::sin(A.Angle + glm::radians(Params.VisionAngle));
    EndPos SensorRight = CalcEndPosition(A.Pos + Dir * Params.VisionDist, A.TriID, Tris, Verts, T2T);
    float Right = Sense(SensorRight.Pos, TrailMap, Width, Height, Params);

    // Look left
    Dir.x = glm::cos(A.Angle - glm::radians(Params.VisionAngle));
    Dir.y = glm::sin(A.Angle - glm::radians(Params.VisionAngle));
    EndPos SensorLeft = CalcEndPosition(A.Pos + Dir * Params.VisionDist, A.TriID, Tris, Verts, T2T);
    float Left = Sense(SensorLeft.Pos, TrailMap, Width, Height, Params);


    // If greater concentration of pheromone is forward, go ahead
    if (Fwd > Right && Fwd > Left)
        return A;
    
    // Determine direction, and add a bit of randomness
    A.RandState = RandHash(A.RandState);
    float RandSteer = ScaleTo01(A.RandState);
    float Turn = Params.TurnSpeed * Params.DeltaTime * 3.14159265;

    // If concentration forward is very small, act randomly
    if (Fwd < Left && Fwd < Right)
        A.Angle += (RandSteer - 0.5f) * 2 * Turn;
    // If left is greater, go left
    else if (Left > Right)
        A.Angle -= RandSteer * Turn;
    // If right is greater, go right
    else
        A.Angle += RandSteer * Turn;

    return A;
}


__global__ void UpdatePositionsKernel(Agent* Agents, float* TrailMap, int Width, int Height, SimulationParameters Params,
                                      mesh::Triangle* Tris, mesh::Vertex* Verts, glm::ivec3* T2T)
{
    int AgentID = blockDim.x * blockIdx.x + threadIdx.x;
    if (AgentID >= Params.NumAgents)
        return;

    // Get the agent and determine direction
    Agent A = Agents[AgentID];
    A = NextDir(A, TrailMap, Width, Height, Verts, Tris, T2T, Params);

    // Compute the end position
    glm::vec2 Dir(glm::cos(A.Angle), glm::sin(A.Angle));
    glm::vec2 Move = Dir * Params.MoveSpeed * Params.DeltaTime;
    EndPos ep = CalcEndPosition(A.Pos + Move, A.TriID, Tris, Verts, T2T);
    EndPos dep = CalcEndPosition(A.Pos + 2.0f * Move, A.TriID, Tris, Verts, T2T);
    A.Pos = ep.Pos;
    // If agent exits from the triangle
    if (A.TriID != ep.TriID)
    {
        // Next direction is determined by the delta between ep and dep, if they are on the same triangle
        if (ep.TriID == dep.TriID)
        {
            glm::vec2 NewDir = dep.Pos - ep.Pos;
            A.Angle = glm::atan(NewDir.y, NewDir.x);
        }
        // Otherwise we take the direction at random
        else
        {
            A.RandState = RandHash(A.RandState);
            A.Angle = ScaleTo01(A.RandState) * 3.14159265;
        }
    }
    A.TriID = ep.TriID;
    Agents[AgentID] = A;

    // Get the agent position in trail map
    int X = (int)(A.Pos.x * (Width - 1));
    int Y = (int)(A.Pos.y * (Height - 1));
    X = glm::clamp(X, 0, Width - 1);
    Y = glm::clamp(Y, 0, Height - 1);
    TrailMap[Y * Width + X] = 1.0f;
}




void slime::SlimeSim3D::LaunchUpdatePositionsKernel()
{
    dim3 bSize(1024);
    dim3 gSize((Params.NumAgents + bSize.x - 1) / bSize.x);
    UpdatePositionsKernel<<<gSize, bSize>>>(dAgents, dTrailMap, TrailMapTex.Width, TrailMapTex.Height, Params, dTris, dVerts, dT2T);
    cudaErrorCheck(cudaDeviceSynchronize());
}