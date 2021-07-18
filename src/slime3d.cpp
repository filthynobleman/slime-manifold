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


void Usage(const char* argv0)
{
    std::cerr << "Bad syntax." << std::endl;
    std::cerr << "Correct syntax for calling " << argv0 << " is" << std::endl;
    std::cerr << "    " << argv0 << " config_file obj_file [ tex_res [mtl_file [light_file]]]" << std::endl;
}


int main(int argc, const char** argv)
{
    try
    {
        if (argc < 3 || argc > 5)
        {
            Usage(argv[0]);
            return -1;
        }

        std::string ConfigFile(argv[1]);
        std::string MeshFile(argv[2]);
        GLuint TexRes = 1024;
        if (argc > 3)
            TexRes = std::atoi(argv[3]);
        std::string MatFile("../data/meshes/default.mtl");
        if (argc > 4)
            MatFile = argv[4];
        std::string LightFile("../data/meshes/default.light");
        if (argc > 5)
            LightFile = argv[5];


        mesh::MeshLoader ML;
        ML.LoadMesh(MeshFile);
        render::Material Material = render::LoadMaterial(MatFile);
        render::Light Light = render::LoadLight(LightFile);
        mesh::Mesh M = ML.GetMesh();

        render::MeshRenderEngine::Init();
        render::MeshRenderEngine::SetCallbacks();
        render::MeshRenderEngine::CreateShader("../data/shaders/Base.vert",
                                               "../data/shaders/Textured.frag");
        // render::Texture Plaster = render::LoadTexture("../data/textures/white_plaster_01_diffuse.png");
        // render::Texture Concrete = render::LoadTexture("../data/textures/green_concrete_pavement_diffuse.png");
        // render::Texture Slime = render::CreateTexture(4096, 4096, GL_RED);
        render::Texture Slime = render::CreateTexture(TexRes, TexRes, GL_RED);
        render::MeshRenderEngine::CreateBuffers(M);
        render::MeshRenderEngine::SetMaterial(Material);
        render::MeshRenderEngine::SetLight(Light);
        // render::MeshRenderEngine::AddTexture("Plaster", Plaster);
        // render::MeshRenderEngine::AddTexture("Concrete", Concrete);
        // render::MeshRenderEngine::AddTexture("Slime", Slime);

        slime::SlimeSim3D SS3D(ConfigFile, M, ML.GetTri2TriAdjMap(), Slime);
        SS3D.InitAgents();

        unsigned int NumFrame = 0;
        while (!render::MeshRenderEngine::ShouldClose())
        {
            // SS3D.UpdatePositions();
            // SS3D.DiffuseTrail();
            SS3D.SimulationStep();
            SS3D.WriteTexture();
            render::MeshRenderEngine::Draw();

            if (render::MeshRenderEngine::MustExport())
                SS3D.ExportFrame();
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return 0;
}