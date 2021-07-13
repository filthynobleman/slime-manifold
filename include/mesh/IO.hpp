/**
 * @file        IO.hpp
 * 
 * @brief       Classes and functions for mesh input/output operations.
 * 
 * @details     
 * 
 * @author      Filippo Maggioli (maggioli@di.uniroma1.it)
 *              Sapienza, University of Rome - Department of Computer Science
 * 
 * @date 2021-07-01
 * 
 */
#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <mesh/Mesh.hpp>

namespace mesh
{
    
class MeshLoader
{
private:
    // Vertex properties
    std::vector<glm::vec3> VPos;
    std::vector<glm::vec3> VNorm;
    std::vector<glm::vec2> VTex;

    // Triangle indices
    std::vector<glm::ivec3> TPos;
    std::vector<glm::ivec3> TNorm;
    std::vector<glm::ivec3> TTex;


public:
    MeshLoader();
    ~MeshLoader();


    void LoadMesh(const std::string& filename);

    Mesh GetMesh() const;

    std::vector<glm::ivec3> GetTri2TriAdjMap() const;
};

} // namespace mesh
