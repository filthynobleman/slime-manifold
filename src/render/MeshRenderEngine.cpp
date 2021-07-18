#include <render/MeshRenderEngine.hpp>

#include <iostream>
#include <fstream>
#include <exception>
#include <sstream>



using namespace render;


void FBCallback(GLFWwindow* Window, int x, int y)
{
    glViewport(0, 0, x, y);
}

void SCallback(GLFWwindow* Window, double x, double y)
{
    render::MeshRenderEngine::ChangeZoom(-y, render::MeshRenderEngine::GetDeltaTime());
}


GLuint render::MeshRenderEngine::VAO            = 0;
GLuint render::MeshRenderEngine::VBO            = 0;
GLuint render::MeshRenderEngine::EBO            = 0;
GLuint render::MeshRenderEngine::NTris          = 0;

GraphicShader* render::MeshRenderEngine::Shader = NULL;

double render::MeshRenderEngine::MouseOldX      = 0.0;
double render::MeshRenderEngine::MouseOldY      = 0.0;
float render::MeshRenderEngine::CurTime         = 0.0f;
float render::MeshRenderEngine::OldTime         = 0.0f;
float render::MeshRenderEngine::LastActionTime  = 0.0f;
float render::MeshRenderEngine::ActionDelay     = 0.1f;

GLenum render::MeshRenderEngine::PolyModes[2]   = { GL_FILL, GL_LINE };
int render::MeshRenderEngine::CurMode           = 0;

bool render::MeshRenderEngine::Export           = false;


std::vector<Texture> render::MeshRenderEngine::Textures;
std::vector<std::string> render::MeshRenderEngine::TexNames;
Material render::MeshRenderEngine::MeshMaterial;
Light render::MeshRenderEngine::SceneLight;


void render::MeshRenderEngine::CreateBuffers(const mesh::Mesh& M)
{
    // Create vertex array and buffers
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Bind and fill vertex buffer
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, M.NVerts() * sizeof(mesh::Vertex), M.Verts.data(), GL_STATIC_DRAW);
    // Bind vertices' positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(mesh::Vertex), (void*)offsetof(mesh::Vertex, Position));
    glEnableVertexAttribArray(0);
    // Bind vertices' normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(mesh::Vertex), (void*)offsetof(mesh::Vertex, Normal));
    glEnableVertexAttribArray(1);
    // Bind vertices' UV
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(mesh::Vertex), (void*)offsetof(mesh::Vertex, TexUV));
    glEnableVertexAttribArray(2);

    // Bind and fill triangles
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, M.NTris() * sizeof(mesh::Triangle), M.Tris.data(), GL_STATIC_DRAW);

    // Save number of triangles
    NTris = M.NTris();

    // Unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}


void render::MeshRenderEngine::CreateShader(const std::string& vert, const std::string& frag)
{
    Shader = new GraphicShader(vert, frag);
    Shader->Compile();
}

void render::MeshRenderEngine::SetLight(const std::string& filename)
{
    SetLight(render::LoadLight(filename));
}

void render::MeshRenderEngine::SetLight(const Light& L)
{
    SceneLight = L;
}

void render::MeshRenderEngine::SetMaterial(const std::string& filename)
{
    SetMaterial(render::LoadMaterial(filename));
}

void render::MeshRenderEngine::SetMaterial(const Material& M)
{
    MeshMaterial = M;
}

void render::MeshRenderEngine::AddTexture(const std::string& TexName, const Texture& Tex)
{
    Textures.push_back(Tex);
    Textures[Textures.size() - 1].BindIdx = Textures.size() - 1;
    TexNames.push_back(TexName);
    assert(Textures.size() == TexNames.size());
}

void render::MeshRenderEngine::SetCallbacks()
{
    SetFramebufferSizeCallback(FBCallback);
    SetScrollCallback(SCallback);
}

void render::MeshRenderEngine::BindAllTextures()
{
    for (GLuint i = 0; i < Textures.size(); ++i)
    {
        Texture Tex = Textures[i];
        glActiveTexture(GL_TEXTURE0 + Tex.BindIdx);
        glBindTexture(GL_TEXTURE_2D, Tex.BufIdx);
        Shader->SetTexture(TexNames[i], Tex);
    }
}


void render::MeshRenderEngine::ProcessInput()
{
    // Close window with button ESC
    if (glfwGetKey(Window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        Close();
        return;
    }

    // Mouse buttons
    double MouseX, MouseY;
    glfwGetCursorPos(Window, &MouseX, &MouseY);
    // Left mouse handles rotation
    if (glfwGetMouseButton(Window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS)
    {
        double X = MouseX - MouseOldX;
        double Y = MouseY - MouseOldY;
        YRotation(X, GetDeltaTime());
        XRotation(-Y, GetDeltaTime());
    }
    // Right button handles scale
    else if (glfwGetMouseButton(Window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS)
    {
        double Y = MouseY - MouseOldY;
        ChangeScale(Y, GetDeltaTime());
    }
    MouseOldX = MouseX;
    MouseOldY = MouseY;

    // Keyboard
    // Ignore any keys until delay is passed
    if (LastActionTime + ActionDelay > glfwGetTime())
        return;

    // Z = wireframe ON/OFF
    if (glfwGetKey(Window, GLFW_KEY_Z) == GLFW_PRESS)
        CurMode = (CurMode + 1) % 2;
    // Space = reset model transform
    if (glfwGetKey(Window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        Pitch   = DefaultPitch;
        Yaw     = DefaultYaw;
        Scale   = DefaultScale;
        // Shift modifier = also reset zoom
        if (glfwGetKey(Window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(Window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
                Zoom = DefaultZoom;
    }
    // E = export texture
    if (glfwGetKey(Window, GLFW_KEY_E) == GLFW_PRESS)
        Export = true;

    // Update action time
    LastActionTime = glfwGetTime();
}

float render::MeshRenderEngine::GetDeltaTime()
{
    // return CurTime - OldTime;
    return 1.0f / 60.0f;
}

bool render::MeshRenderEngine::MustExport()
{
    if (Export)
    {
        Export = false;
        return true;
    }
    return false;
}


void render::MeshRenderEngine::Draw()
{
    // Clear window
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(BGCol.r, BGCol.g, BGCol.b, 1.0f);

    glPolygonMode(GL_FRONT_AND_BACK, PolyModes[CurMode]);

    // Compute transformation matrices
    glm::mat4 Model         = GetModelMatrix();
    glm::mat4 View          = GetViewMatrix();
    glm::mat4 Projection    = GetProjectionMatrix();
    glm::mat3 ModelInv      = glm::mat3(glm::inverse(glm::transpose(Model)));

    // Use shader
    Shader->Use();

    // Set shader's properties
    // Material and light
    Shader->SetMaterial("Material", MeshMaterial);
    Shader->SetLight("Light",       SceneLight);
    // Coordinates transformation
    Shader->SetMat4("Model",        Model);
    Shader->SetMat3("ModelInv",     ModelInv);
    Shader->SetMat4("View",         View);
    Shader->SetMat4("Projection",   Projection);
    // Camera position
    Shader->SetVec3("CameraPos",    CameraPos);
    
    // Bind textures
    BindAllTextures();

    // Draw the scene
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 3 * NTris, GL_UNSIGNED_INT, (void*)0);
    glBindVertexArray(0);

    // Swap buffers and poll input events
    glfwSwapBuffers(Window);
    glfwPollEvents();
    ProcessInput();

    // Update times
    OldTime = CurTime;
    CurTime = glfwGetTime();
}