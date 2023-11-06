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
#include <chrono>


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
    std::cerr << "    " << argv0 << " config_file obj_file [ tex_res [export_video [mtl_file [light_file]]]]" << std::endl;
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
        GLuint TexRes = 4096;
        bool ExportVideo = false;
        if (argc > 3)
            TexRes = std::atoi(argv[3]);
        if (argc > 4)
            ExportVideo = std::atoi(argv[4]) > 0;
        std::string MatFile("../data/meshes/default.mtl");
        if (argc > 5)
            MatFile = argv[5];
        std::string LightFile("../data/meshes/default.light");
        if (argc > 6)
            LightFile = argv[6];


        mesh::MeshLoader ML;
        ML.LoadMesh(MeshFile);
        render::Material Material = render::LoadMaterial(MatFile);
        render::Light Light = render::LoadLight(LightFile);
        mesh::Mesh M = ML.GetMesh();

        render::MeshRenderEngine::Init();
        render::MeshRenderEngine::SetCallbacks();
        render::MeshRenderEngine::CreateShader("../data/shaders/Base.vert",
                                               "../data/shaders/Textured.frag");
                                               
        slime::SimulationParameters Params = slime::LoadFromFile(ConfigFile);
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
        render::Texture Slime = render::CreateTexture(TexRes, TexRes, Format);
        render::MeshRenderEngine::CreateBuffers(M);
        render::MeshRenderEngine::SetMaterial(Material);
        render::MeshRenderEngine::SetLight(Light);
        render::MeshRenderEngine::AddTexture("NoiseTex", Slime);

        slime::SlimeSim3D SS3D(ConfigFile, M, ML.GetTri2TriAdjMap(), Slime);
        SS3D.InitAgents();

        unsigned int NumFrame = 0;
        float AvgTPF = 0.0f;
        float MaxTPF = -INFINITY;
        float MinTPF = INFINITY;
        glfwSwapInterval(0);
        while (!render::MeshRenderEngine::ShouldClose())
        {
            std::chrono::system_clock::time_point Start = std::chrono::system_clock::now();

            SS3D.SimulationStep();
            SS3D.WriteTexture();
            render::MeshRenderEngine::Draw();

            if (render::MeshRenderEngine::MustExport() || ExportVideo)
                SS3D.ExportFrame();

            std::chrono::system_clock::time_point End = std::chrono::system_clock::now();
            float TPF = std::chrono::duration_cast<std::chrono::nanoseconds>(End - Start).count();
            AvgTPF += TPF;
            MaxTPF = glm::max(MaxTPF, TPF);
            MinTPF = glm::min(MinTPF, TPF);
            NumFrame += 1;
        }

        AvgTPF /= NumFrame;
        std::cout << "Average TPF: " << AvgTPF / 1e6f << std::endl;
        std::cout << "Maximum TPF: " << MaxTPF / 1e6f << std::endl;
        std::cout << "Minimum TPF: " << MinTPF / 1e6f << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return 0;
}