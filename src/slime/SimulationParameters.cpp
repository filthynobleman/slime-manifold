#include <slime/SimulationParameters.hpp>


using namespace slime;

#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>

SimulationParameters slime::LoadFromFile(const std::string& Filename)
{
    SimulationParameters Params;
    Params.NumSpecies = 1;

    std::ifstream Stream;
    Stream.open(Filename, std::ios::in);

    if (!Stream.is_open())
    {
        std::stringstream ss;
        ss << "Cannot open file " << Filename << " for reading.";
        throw std::exception(ss.str().c_str());
    }

    std::string Line;
    while (!Stream.eof())
    {
        std::getline(Stream, Line);

        if (Line.rfind("NumAgents", 0) == 0)
        {
            std::sscanf(Line.c_str(), "NumAgents=%d", &(Params.NumAgents));
            Params.AgentGridSize = (int)ceil(sqrt(Params.NumAgents));
        }
        else if (Line.rfind("NumSpecies", 0) == 0)
            std::sscanf(Line.c_str(), "NumSpecies=%d", &(Params.NumSpecies));

        else if (Line.rfind("MoveSpeed", 0) == 0)
            std::sscanf(Line.c_str(), "MoveSpeed=%f", &(Params.MoveSpeed));
        else if (Line.rfind("TurnSpeed", 0) == 0)
            std::sscanf(Line.c_str(), "TurnSpeed=%f", &(Params.TurnSpeed));
        else if (Line.rfind("MoveStep", 0) == 0)
            std::sscanf(Line.c_str(), "MoveStep=%f", &(Params.MoveStep));
            
        else if (Line.rfind("VisionAngle", 0) == 0)
            std::sscanf(Line.c_str(), "VisionAngle=%f", &(Params.VisionAngle));
        else if (Line.rfind("VisionDist", 0) == 0)
            std::sscanf(Line.c_str(), "VisionDist=%f", &(Params.VisionDist));
        else if (Line.rfind("SensorRadius", 0) == 0)
            std::sscanf(Line.c_str(), "SensorRadius=%d", &(Params.SensorRadius));
            
        else if (Line.rfind("DecayRate", 0) == 0)
            std::sscanf(Line.c_str(), "DecayRate=%f", &(Params.DecayRate));
        else if (Line.rfind("DiffuseRate", 0) == 0)
            std::sscanf(Line.c_str(), "DiffuseRate=%f", &(Params.DiffuseRate));
        else if (Line.rfind("TrailWeight", 0) == 0)
            std::sscanf(Line.c_str(), "TrailWeight=%f", &(Params.TrailWeight));
            
        else if (Line.rfind("StepsPerFrame", 0) == 0)
            std::sscanf(Line.c_str(), "StepsPerFrame=%d", &(Params.StepsPerFrame));
    }

    Stream.close();

    Params.Time = 0.0f;
    Params.DeltaTime = 0.0f;
    
    if (Params.NumSpecies > 3)
    {
        std::stringstream ss;
        ss << "You cannot have more than three species. Requested " << Params.NumSpecies << " species.";
        throw std::exception(ss.str().c_str());
    }

    return Params;
}


void slime::InitWithMesh(SimulationParameters& Params, const mesh::Mesh& Mesh)
{
    Params.NVerts = Mesh.NVerts();
    Params.NEdges = Mesh.NEdges();
    Params.NTris = Mesh.NTris();
}