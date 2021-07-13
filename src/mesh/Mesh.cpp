#include <mesh/Mesh.hpp>


using namespace mesh;


// Vertex
void InitVertex(Vertex& v, const glm::vec3& XYZ, const glm::vec3& N, const glm::vec2& UV)
{
    v.Position = XYZ;
    v.Normal = N;
    v.TexUV = UV;
}

mesh::Vertex::Vertex(const glm::vec3& XYZ, const glm::vec3& N, const glm::vec2& UV)
{
    InitVertex(*this, XYZ, N, UV);
}


mesh::Vertex::Vertex(const Vertex& v)
{
    InitVertex(*this, v.Position, v.Normal, v.TexUV);
}

Vertex& mesh::Vertex::operator=(const Vertex& v)
{
    InitVertex(*this, v.Position, v.Normal, v.TexUV);
    return *this;
}

mesh::Vertex::~Vertex() {}


float mesh::Vertex::operator[](int i) const
{
    return Position[i];
}
float& mesh::Vertex::operator[](int i)
{
    return Position[i];
}

float mesh::Vertex::operator()(int i) const
{
    return TexUV[i];
}
float& mesh::Vertex::operator()(int i)
{
    return TexUV[i];
}











// Edge
void InitEdge(Edge& e, int i, int j)
{
    e.Verts = glm::ivec2(i, j);
}

mesh::Edge::Edge(const glm::ivec2& vs)
{
    InitEdge(*this, vs[0], vs[1]);
}

mesh::Edge::Edge(int i, int j)
{
    InitEdge(*this, i, j);
}

mesh::Edge::Edge(const Edge& e)
{
    InitEdge(*this, e[0], e[1]);
}

Edge& mesh::Edge::operator=(const Edge& e)
{
    InitEdge(*this, e[0], e[1]);
    return *this;
}

mesh::Edge::~Edge() {}

int mesh::Edge::operator[](int i) const
{
    return Verts[i];
}
int& mesh::Edge::operator[](int i)
{
    return Verts[i];
}

bool mesh::operator==(const Edge& e1, const Edge& e2)
{
    return (e1[0] == e2[0] && e1[1] == e2[1]) ||
           (e1[0] == e2[1] && e1[1] == e2[0]);
}

bool mesh::operator!=(const Edge& e1, const Edge& e2)
{
    return !(e1 == e2);
}

std::size_t std::hash<Edge>::operator()(const Edge& e) const noexcept
{
    std::size_t h1 = std::hash<int>{}(e[0]);
    std::size_t h2 = std::hash<int>{}(e[1]);
    return (h1 ^ (h2 << 1)) + (h2 ^ (h1 << 1));
}








// Triangle
void InitTriangle(Triangle& t, int i, int j, int k)
{
    t.Verts = glm::ivec3(i, j, k);
}

mesh::Triangle::Triangle(const glm::ivec3& vs)
{
    InitTriangle(*this, vs[0], vs[1], vs[2]);
}

mesh::Triangle::Triangle(int i, int j, int k)
{
    InitTriangle(*this, i, j, k);
}

mesh::Triangle::Triangle(const Triangle& t)
{
    InitTriangle(*this, t[0], t[1], t[2]);
}

Triangle& mesh::Triangle::operator=(const Triangle& t)
{
    InitTriangle(*this, t[0], t[1], t[2]);
    return *this;
}

mesh::Triangle::~Triangle() {}

int mesh::Triangle::operator[](int i) const
{
    return Verts[i];
}
int& mesh::Triangle::operator[](int i)
{
    return Verts[i];
}

bool mesh::operator==(const Triangle& t1, const Triangle& t2)
{
    int i;
    for (int off = 0; off < 3; ++off)
    {
        for (i = 0; i < 3; ++i)
        {
            if (t1[i] != t2[(i + off) % 3])
                break;
        }
        if (i == 3)
            return true;
    }
    return false;
}

bool mesh::operator!=(const Triangle& t1, const Triangle& t2)
{
    return !(t1 == t2);
}

std::size_t std::hash<Triangle>::operator()(const Triangle& t) const noexcept
{
    std::size_t h[3];
    for (int i = 0; i < 3; ++i)
        h[i] = std::hash<int>{}(t[i]);
    std::size_t H = 0;
    for (int i = 0; i < 3; ++i)
        H += h[i] ^ (h[(i + 1) % 3] << 1);
    return H;
}












// Mesh
void InitMesh(Mesh& M, const std::vector<Vertex>& Vs, const std::vector<Edge>& Es, const std::vector<Triangle>& Ts, bool Reset)
{
    if (Reset)
    {
        M.Verts.clear();
        M.Edges.clear();
        M.Tris.clear();
    }

    M.Verts.insert(M.Verts.begin(), Vs.begin(), Vs.end());
    M.Edges.insert(M.Edges.begin(), Es.begin(), Es.end());
    M.Tris.insert(M.Tris.begin(), Ts.begin(), Ts.end());
}

mesh::Mesh::Mesh() {}

mesh::Mesh::Mesh(const std::vector<Vertex>& Vs, const std::vector<Edge>& Es, const std::vector<Triangle>& Ts)
{
    InitMesh(*this, Vs, Es, Ts, false);
}

mesh::Mesh::Mesh(const Mesh& M)
{
    InitMesh(*this, M.Verts, M.Edges, M.Tris, false);
}

Mesh& mesh::Mesh::operator=(const Mesh& M)
{
    InitMesh(*this, M.Verts, M.Edges, M.Tris, true);
    return *this;
}

mesh::Mesh::~Mesh() {}


int mesh::Mesh::NVerts() const { return Verts.size(); }
int mesh::Mesh::NEdges() const { return Edges.size(); }
int mesh::Mesh::NTris() const { return Tris.size(); }


Vertex mesh::Mesh::operator[](int i) const
{
    return Verts[i];
}
Vertex& mesh::Mesh::operator[](int i)
{
    return Verts[i];
}

Triangle mesh::Mesh::operator()(int i) const
{
    return Tris[i];
}
Triangle& mesh::Mesh::operator()(int i)
{
    return Tris[i];
}