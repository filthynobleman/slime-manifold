/**
 * @file        Mesh.hpp
 * 
 * @brief       Classes for vertices, edges, triangles and meshes.
 * 
 * @author      Filippo Maggioli\n 
 *              (maggioli@di.uniroma1.it, maggioli.filippo@gmail.com)\n 
 *              Sapienza, University of Rome - Department of Computer Science
 * @date        2021-06-30
 */
#pragma once

#include <glm/glm.hpp>
#include <vector>

#include <cuda_runtime.h>

namespace mesh
{
    
struct Vertex
{
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexUV;

    Vertex(const glm::vec3& XYZ, 
           const glm::vec3& N, 
           const glm::vec2& UV);
    Vertex(const Vertex& v);
    Vertex& operator=(const Vertex& v);
    ~Vertex();

    float operator[](int i) const;
    float& operator[](int i);

    float operator()(int i) const;
    float& operator()(int i);
};


struct Edge
{
    glm::ivec2 Verts;

    Edge(const glm::ivec2& Vs);
    Edge(int i, int j);
    Edge(const Edge& e);
    Edge& operator=(const Edge& e);
    ~Edge();

    int operator[](int i) const;
    int& operator[](int i);
};

bool operator==(const Edge& e1, const Edge& e2);
bool operator!=(const Edge& e1, const Edge& e2);


struct Triangle
{
    glm::ivec3 Verts;

    Triangle(const glm::ivec3& Vs);
    Triangle(int i, int j, int k);
    Triangle(const Triangle& t);
    Triangle& operator=(const Triangle& t);
    ~Triangle();

    int operator[](int i) const;
    int& operator[](int i);

    //glm::vec3 Normal(const Mesh& M) const;
};

bool operator==(const Triangle& t1, const Triangle& t2);
bool operator!=(const Triangle& t1, const Triangle& t2);



struct Mesh
{
    std::vector<Vertex>     Verts;
    std::vector<Edge>       Edges;
    std::vector<Triangle>   Tris;

    Mesh();
    Mesh(const std::vector<Vertex>& Vs,
         const std::vector<Edge>& Es,
         const std::vector<Triangle>& Ts);
    Mesh(const Mesh& mesh);
    Mesh& operator=(const Mesh& mesh);
    ~Mesh();

    int NVerts() const;
    int NEdges() const;
    int NTris() const;

    Vertex operator[](int i) const;
    Vertex& operator[](int i);

    Triangle operator()(int i) const;
    Triangle& operator()(int i);

    std::vector<glm::mat3> UVTo3DRescale() const;
};


} // namespace mesh


namespace std
{
    template<>
    struct hash<mesh::Edge>
    {
        std::size_t operator()(const mesh::Edge& e) const noexcept;
    };
    
    template<>
    struct hash<mesh::Triangle>
    {
        std::size_t operator()(const mesh::Triangle& t) const noexcept;
    };
}

