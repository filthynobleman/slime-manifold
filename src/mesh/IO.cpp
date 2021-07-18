#include <mesh/IO.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>
#include <unordered_map>


using namespace mesh;


mesh::MeshLoader::MeshLoader() {}
mesh::MeshLoader::~MeshLoader() {}


void mesh::MeshLoader::LoadMesh(const std::string& filename)
{
    std::ifstream stream;
    stream.open(filename, std::ios::in);

    if (!stream.is_open())
    {
        std::stringstream ss;
        ss << "Cannot open file " << filename << " for reading.";
        throw std::exception(ss.str().c_str());
    }


    std::string Line;
    while (!stream.eof())
    {
        std::getline(stream, Line);

        if (Line.rfind("v ", 0) == 0)
        {
            glm::vec3 v;
            std::sscanf(Line.c_str(), "v %f %f %f", 
                        &(v.x), &(v.y), &(v.z));
            VPos.push_back(v);
        }
        else if (Line.rfind("vn ", 0) == 0)
        {
            glm::vec3 vn;
            std::sscanf(Line.c_str(), "vn %f %f %f", 
                        &(vn.x), &(vn.y), &(vn.z));
            VNorm.push_back(vn);
        }
        else if (Line.rfind("vt ", 0) == 0)
        {
            glm::vec2 vt;
            std::sscanf(Line.c_str(), "vt %f %f",
                        &(vt.x), &(vt.y));
            VTex.push_back(vt);
        }
        else if (Line.rfind("f ", 0) == 0)
        {
            glm::imat3x3 f;
            std::sscanf(Line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d",
                        &(f[0][0]), &(f[1][0]), &(f[2][0]),
                        &(f[0][1]), &(f[1][1]), &(f[2][1]),
                        &(f[0][2]), &(f[1][2]), &(f[2][2]));
            glm::ivec3 one(1, 1, 1);
            TPos.push_back(f[0] - one);
            TTex.push_back(f[1] - one);
            TNorm.push_back(f[2] - one);
        }
    }


    stream.close();
}

Mesh mesh::MeshLoader::GetMesh() const
{
    // Create vertex map and update triangles' indices
    std::unordered_map<glm::ivec3, int> VMap;
    std::vector<Triangle>               T;
    int SerialVertID = 0;
    for (int i = 0; i < TPos.size(); ++i)
    {
        glm::ivec3 TFix;
        for (int j = 0; j < 3; ++j)
        {
            glm::ivec3 VID(TPos[i][j], TTex[i][j], TNorm[i][j]);
            if (VMap.find(VID) == VMap.end())
                VMap[VID] = SerialVertID++;
            TFix[j] = VMap[VID];
        }
        T.push_back(TFix);
    }

    // Create the edge map
    std::unordered_map<Edge, int> EMap;
    int SerialEdgeID = 0;
    for (int i = 0; i < T.size(); ++i)
    {
        Triangle t = T[i];
        for (int j = 0; j < 3; ++j)
        {
            Edge e(t[j], t[(j + 1) % 3]);
            if (EMap.find(e) == EMap.end())
                EMap[e] = SerialEdgeID++;
        }
    }

    // Create vertex vector
    Vertex* _V = (Vertex*)calloc(VMap.size(), sizeof(Vertex));
    if (_V == NULL)
        throw std::exception("Cannot allocate vertices.");
    std::unordered_map<glm::ivec3, int>::iterator vit;
    for (vit = VMap.begin(); vit != VMap.end(); vit++)
    {
        int i = vit->second;
        glm::ivec3 VIDs = vit->first;
        _V[i].Position = VPos[VIDs[0]];
        _V[i].TexUV = VTex[VIDs[1]];
        _V[i].Normal = VNorm[VIDs[2]];
    }
    std::vector<Vertex> V(_V, _V + VMap.size());
    free(_V);
    
    // Create edge vector
    Edge* _E = (Edge*)calloc(EMap.size(), sizeof(Edge));
    if (_E == NULL)
        throw std::exception("Cannot allocate edges");
    std::unordered_map<Edge, int>::iterator eit;
    for (eit = EMap.begin(); eit != EMap.end(); eit++)
        _E[eit->second] = eit->first;
    std::vector<Edge> E(_E, _E + EMap.size());
    free(_E);


    // Create the mesh and return
    return Mesh(V, E, T);
}

std::vector<glm::ivec3> mesh::MeshLoader::GetTri2TriAdjMap() const
{
    // Create a map from edges to adjacent triangles
    std::unordered_map<Edge, glm::ivec2> E2T;
    for (int i = 0; i < TPos.size(); ++i)
    {
        glm::ivec3 t = TPos[i];
        for (int j = 0; j < 3; ++j)
        {
            Edge e(t[j], t[(j + 1) % 3]);
            if (E2T.find(e) == E2T.end())
                E2T[e] = glm::ivec2(-1, -1);
            glm::ivec2 El = E2T[e];
            if (El.x == -1) El.x = i;
            else            El.y = i;
            E2T[e] = El;
        }
    }

    // Initialize T2T map
    std::vector<glm::ivec3> T2T;
    for (int i = 0; i < TPos.size(); ++i)
        T2T.push_back(glm::ivec3(-1, -1, -1));

    // Fill T2T map
    std::unordered_map<Edge, glm::ivec2>::iterator Eit;
    for (Eit = E2T.begin(); Eit != E2T.end(); Eit++)
    {
        glm::ivec2 Ts = (*Eit).second;
        for (int j = 0; j < 3; ++j)
        {
            Edge Opp(TPos[Ts[0]][(j + 1) % 3], TPos[Ts[0]][(j + 2) % 3]);
            if (Eit->first == Opp)
            {
                T2T[Ts[0]][j] = Ts[1];
                break;
            }
        }

        if (Ts[1] < 0)
            continue;
            
        for (int j = 0; j < 3; ++j)
        {
            Edge Opp(TPos[Ts[1]][(j + 1) % 3], TPos[Ts[1]][(j + 2) % 3]);
            if (Eit->first == Opp)
            {
                T2T[Ts[1]][j] = Ts[0];
                break;
            }
        }
    }

    return T2T;
}