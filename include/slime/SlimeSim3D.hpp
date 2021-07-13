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

namespace slime
{
    
struct Agent
{
    glm::vec2       Pos;
    float           Angle;
    int             TriID;
    unsigned int    RandState;
};


class SlimeSim3D
{
private:
    Agent*                          dAgents;
    glm::ivec3*                     dT2T;
    mesh::Vertex*                   dVerts;
    mesh::Triangle*                 dTris;
    float*                          dTrailMap;
    float*                          dDiffuseTrail;
    GLuint                          PBO;
    SimulationParameters            Params;
    const render::Texture           TrailMapTex;
    int                             NVerts;
    int                             NTris;

    void LaunchInitAgentsKernel();
    void LaunchUpdatePositionsKernel();
    void LaunchDiffuseTrailKernel();
    

public:
    SlimeSim3D(const std::string& ParamsFile, const mesh::Mesh& Mesh, 
               const std::vector<glm::ivec3>& T2T, const render::Texture& TMTex);
    ~SlimeSim3D();

    void InitAgents();
    void UpdatePositions();
    void DiffuseTrail();
    void WriteTexture();
};


} // namespace slime
