#include <slime/SlimeSim3D.hpp>
#include <render/MeshRenderEngine.hpp>


using namespace slime;


slime::SlimeSim3D::SlimeSim3D(const std::string& ParamsFile, const mesh::Mesh& Mesh, 
                              const std::vector<glm::ivec3>& T2T, const render::Texture &TMTex)
    : TrailMapTex(TMTex), NVerts(Mesh.NVerts()), NTris(Mesh.NTris())
{
    // Load parameters
    Params = slime::LoadFromFile(ParamsFile);
    slime::InitWithMesh(Params, Mesh);

    // Allocate GPU memory and copy
    cudaErrorCheck(cudaCalloc<Agent>(&dAgents, Params.NumAgents));
    cudaErrorCheck(cudaAllocCopy<mesh::Vertex>(&dVerts, Mesh.Verts.data(), NVerts));
    cudaErrorCheck(cudaAllocCopy<mesh::Triangle>(&dTris, Mesh.Tris.data(), NTris));
    cudaErrorCheck(cudaAllocCopy<glm::ivec3>(&dT2T, T2T.data(), NTris));
    cudaErrorCheck(cudaCalloc<float>(&dTrailMap, TMTex.Width * TMTex.Height));

    // Create the pixul buffer object
    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, TrailMapTex.Width * TrailMapTex.Height * sizeof(float), NULL, GL_DYNAMIC_COPY);
    cudaErrorCheck(cudaGLRegisterBufferObject(PBO));
}


slime::SlimeSim3D::~SlimeSim3D()
{
    cudaErrorCheck(cudaGLUnregisterBufferObject(PBO));
    glDeleteBuffers(1, &PBO);
    cudaErrorCheck(cudaFree(dAgents));
    cudaErrorCheck(cudaFree(dVerts));
    cudaErrorCheck(cudaFree(dTris));
    cudaErrorCheck(cudaFree(dT2T));
    cudaErrorCheck(cudaFree(dTrailMap));
}


void slime::SlimeSim3D::InitAgents()
{
    LaunchInitAgentsKernel();
}

void slime::SlimeSim3D::UpdatePositions()
{
    Params.DeltaTime = render::MeshRenderEngine::GetDeltaTime();
    Params.Time += Params.DeltaTime;
    LaunchUpdatePositionsKernel();
}

void slime::SlimeSim3D::DiffuseTrail()
{
    cudaErrorCheck(cudaGLMapBufferObject((void**)&dDiffuseTrail, PBO));
    LaunchDiffuseTrailKernel();
    cudaErrorCheck(cudaMemcpy(dTrailMap, dDiffuseTrail, 
                              TrailMapTex.Width * TrailMapTex.Height * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
    cudaErrorCheck(cudaGLUnmapBufferObject(PBO));
}

void slime::SlimeSim3D::WriteTexture()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBindTexture(GL_TEXTURE_2D, TrailMapTex.BufIdx);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TrailMapTex.Width, TrailMapTex.Height, GL_RED, GL_FLOAT, NULL);
}