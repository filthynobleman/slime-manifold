#include <slime/SlimeSim3D.hpp>
#include <render/MeshRenderEngine.hpp>


using namespace slime;


slime::SlimeSim3D::SlimeSim3D(const std::string& ParamsFile, const mesh::Mesh& Mesh, 
                              const std::vector<glm::ivec3>& T2T, const render::Texture &TMTex)
    : TrailMapTex(TMTex), StaticTex(TMTex), NVerts(Mesh.NVerts()), NTris(Mesh.NTris())
{
    // Load parameters
    Params = slime::LoadFromFile(ParamsFile);
    slime::InitWithMesh(Params, Mesh);

    // Allocate GPU memory and copy
    cudaErrorCheck(cudaCalloc<Agent>(&dAgents, Params.NumAgents));
    cudaErrorCheck(cudaAllocCopy<mesh::Vertex>(&dVerts, Mesh.Verts.data(), NVerts));
    cudaErrorCheck(cudaAllocCopy<mesh::Triangle>(&dTris, Mesh.Tris.data(), NTris));
    cudaErrorCheck(cudaAllocCopy<glm::ivec3>(&dT2T, T2T.data(), NTris));
    cudaErrorCheck(cudaAllocCopy<float>(&UVTo3D, Mesh.UVTo3DRescale().data(), Mesh.NTris()));
    cudaErrorCheck(cudaCalloc<float>(&dTrailMap, TMTex.Width * TMTex.Height * Params.NumSpecies));
    dStaticTrail = NULL;

    // Create the pixel buffer object
    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, TrailMapTex.Width * TrailMapTex.Height * Params.NumSpecies * sizeof(float), NULL, GL_DYNAMIC_COPY);
    cudaErrorCheck(cudaGLRegisterBufferObject(PBO));

    // Create slime exporter
    Exporter = new SlimeExporter("output", TMTex);
}

slime::SlimeSim3D::SlimeSim3D(const std::string& ParamsFile, const mesh::Mesh& Mesh, 
                              const std::vector<glm::ivec3>& T2T, const render::Texture &TMTex,
                              const unsigned char* StaticTrail, const render::Texture& StaticTex,
                              float ObstacleWeight, float AttractorWeight)
    : TrailMapTex(TMTex), StaticTex(StaticTex), NVerts(Mesh.NVerts()), NTris(Mesh.NTris()), 
      ObstacleWeight(ObstacleWeight), AttractorWeight(AttractorWeight)
{
    assert(StaticTex.Width == TMTex.Width);
    assert(StaticTex.Height == TMTex.Height);

    // Load parameters
    Params = slime::LoadFromFile(ParamsFile);
    slime::InitWithMesh(Params, Mesh);

    // Allocate GPU memory and copy
    cudaErrorCheck(cudaCalloc<Agent>(&dAgents, Params.NumAgents));
    cudaErrorCheck(cudaAllocCopy<mesh::Vertex>(&dVerts, Mesh.Verts.data(), NVerts));
    cudaErrorCheck(cudaAllocCopy<mesh::Triangle>(&dTris, Mesh.Tris.data(), NTris));
    cudaErrorCheck(cudaAllocCopy<glm::ivec3>(&dT2T, T2T.data(), NTris));
    cudaErrorCheck(cudaAllocCopy<float>(&UVTo3D, Mesh.UVTo3DRescale().data(), Mesh.NTris()));
    cudaErrorCheck(cudaCalloc<float>(&dTrailMap, TMTex.Width * TMTex.Height * Params.NumSpecies));
    cudaErrorCheck(cudaAllocCopy<unsigned char>(&dStaticTrail, StaticTrail, TMTex.Width * TMTex.Height * 3));

    // Create the pixel buffer object
    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, TrailMapTex.Width * TrailMapTex.Height * Params.NumSpecies * sizeof(float), NULL, GL_DYNAMIC_COPY);
    cudaErrorCheck(cudaGLRegisterBufferObject(PBO));

    // Create slime exporter
    Exporter = new SlimeExporter("output", TMTex);
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
    if (dStaticTrail != NULL)
        cudaErrorCheck(cudaFree(dStaticTrail));
    if (Exporter != NULL)
        delete Exporter;
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
                              TrailMapTex.Width * TrailMapTex.Height * Params.NumSpecies * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
    cudaErrorCheck(cudaGLUnmapBufferObject(PBO));
}

void slime::SlimeSim3D::SimulationStep()
{
    for (int i = 0; i < Params.StepsPerFrame; ++i)
    {
        UpdatePositions();
        DiffuseTrail();
    }
}

void slime::SlimeSim3D::WriteTexture()
{
    GLenum Format;
    switch (Params.NumSpecies)
    {
    case 1:
        Format = GL_RED;
        break;
    case 2:
        Format = GL_RG;
        break;
    case 3:
        Format = GL_RGB;
        break;
    
    default:
        throw std::exception("Invalid number of species.");
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBindTexture(GL_TEXTURE_2D, TrailMapTex.BufIdx);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TrailMapTex.Width, TrailMapTex.Height, Format, GL_FLOAT, NULL);
}

void slime::SlimeSim3D::ExportFrame()
{
    Exporter->ExportFrame(dTrailMap);
}