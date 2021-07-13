#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <mesh/Mesh.hpp>
#include <mesh/IO.hpp>
#include <render/MeshRenderEngine.hpp>
#include <slime/SlimeSim3D.hpp>

#include <cuda_gl_interop.h>

#include <utils/CUDAUtils.h>


std::ostream& operator<<(std::ostream& os, const glm::vec3& v)
{
    os << v.x << ", " << v.y << ", " << v.z;
    return os;
}
std::ostream& operator<<(std::ostream& os, const glm::vec2& v)
{
    os << v.x << ", " << v.y;
    return os;
}


int main(int argc, const char** argv)
{
    try
    {
        mesh::MeshLoader ML;
        ML.LoadMesh("../data/meshes/icosphere.obj");
        render::Material Material = render::LoadMaterial("../data/meshes/icosphere.mtl");
        render::Light Light = render::LoadLight("../data/meshes/default.light");
        mesh::Mesh M = ML.GetMesh();

        render::MeshRenderEngine::Init();
        render::MeshRenderEngine::SetCallbacks();
        render::MeshRenderEngine::CreateShader("../data/shaders/Base.vert",
                                               "../data/shaders/Textured.frag");
        render::Texture Tex = render::LoadTexture("../data/textures/grass.jpg");
        // render::Texture NoiseTex = render::LoadTexture("../data/textures/Worley.jpg");
        render::Texture NoiseTex = render::CreateTexture(4096, 4096, GL_RED);
        render::MeshRenderEngine::CreateBuffers(M);
        render::MeshRenderEngine::SetMaterial(Material);
        render::MeshRenderEngine::SetLight(Light);
        // render::MeshRenderEngine::AddTexture("Tex", Tex);
        render::MeshRenderEngine::AddTexture("NoiseTex", NoiseTex);

        slime::SlimeSim3D SS3D("../data/configs/Test000.ini", M, ML.GetTri2TriAdjMap(), NoiseTex);
        SS3D.InitAgents();

        while (!render::MeshRenderEngine::ShouldClose())
        {
            SS3D.UpdatePositions();
            SS3D.DiffuseTrail();
            SS3D.WriteTexture();
            render::MeshRenderEngine::Draw();
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return 0;
}