/**
 * @file        SlimeSim3D.hpp
 * 
 * @brief       Interface for CUDA calls handling 3D slime simulation
 * 
 * @details     
 * 
 * @author      Filippo Maggioli (maggioli@di.uniroma1.it)
 *              Sapienza, University of Rome - Department of Computer Science
 * 
 * @date 2021-07-13
 * 
 */
#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <utils/CUDAUtils.h>
#include <mesh/Mesh.hpp>
#include <slime/SimulationParameters.hpp>
#include <render/Material.hpp>
#include <slime/SlimeExporter.hpp>

namespace slime
{
    
struct Agent
{
    glm::vec2       Pos;
    float           Angle;
    int             TriID;
    int             SpeciesID;
    unsigned int    RandState;
};


class SlimeSim3D
{
private:
    Agent*                          dAgents;
    glm::ivec3*                     dT2T;
    mesh::Vertex*                   dVerts;
    mesh::Triangle*                 dTris;
    float*                          UVTo3D;
    float*                          dTrailMap;
    unsigned char*                  dObstacle;
    float*                          dDiffuseTrail;
    GLuint                          PBO;
    SimulationParameters            Params;
    const render::Texture           TrailMapTex;
    const render::Texture           ObstacleTex;
    int                             NVerts;
    int                             NTris;
    bool                            IsObstacle;
    SlimeExporter*                  Exporter;

    void LaunchInitAgentsKernel();
    void LaunchUpdatePositionsKernel();
    void LaunchDiffuseTrailKernel();
    

public:
    SlimeSim3D(const std::string& ParamsFile, const mesh::Mesh& Mesh, 
               const std::vector<glm::ivec3>& T2T, const render::Texture& TMTex);
    SlimeSim3D(const std::string& ParamsFile, const mesh::Mesh& Mesh, 
               const std::vector<glm::ivec3>& T2T, const render::Texture& TMTex,
               const unsigned char* Obstacle, const render::Texture& ObstacleTex,
               bool IsObstacle = true);
    ~SlimeSim3D();

    void InitAgents();
    void SimulationStep();
    void UpdatePositions();
    void DiffuseTrail();
    void WriteTexture();
    void ExportFrame();
};


} // namespace slime
