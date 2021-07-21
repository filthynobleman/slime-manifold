/**
 * @file        SimulationParameters.hpp
 * 
 * @brief       Structure holding the parameters for the slime mold simulation.
 * 
 * @details     Structure holding the parameters for the slime mold simulation. Parameters are the same for both 2D and 3D.
 * 
 * @author      Filippo Maggioli (maggioli@di.uniroma1.it)
 *              Sapienza, University of Rome - Department of Computer Science
 * 
 * @date 2021-07-12
 * 
 */
#pragma once

#include <mesh/Mesh.hpp>
#include <string>

namespace slime
{
    
struct SimulationParameters
{
    int         NVerts;
    int         NEdges;
    int         NTris;

    int         NumAgents;
    int         AgentGridSize;
    int         NumSpecies;

    float       MoveSpeed;
    float       MoveStep;
    float       TurnSpeed;

    float       VisionAngle;
    float       VisionDist;
    int         SensorRadius;

    float       Time;
    float       DeltaTime;
    int         StepsPerFrame;

    float       DecayRate;
    float       DiffuseRate;
    float       TrailWeight;
};


SimulationParameters LoadFromFile(const std::string& Filename);
void InitWithMesh(SimulationParameters& Params, const mesh::Mesh& Mesh);


} // namespace slime
