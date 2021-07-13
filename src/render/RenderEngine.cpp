#include <render/RenderEngine.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace render;


GLFWwindow* render::RenderEngine::Window            = NULL;
int render::RenderEngine::WinWidth                  = 0;
int render::RenderEngine::WinHeight                 = 0;

const float render::RenderEngine::DefaultPitch      = 0.0f;
const float render::RenderEngine::DefaultYaw        = 0.0f;
const float render::RenderEngine::DefaultFoV        = 60.0f;
const float render::RenderEngine::DefaultZoom       = 1.0f;
const float render::RenderEngine::DefaultScale      = 1.0f;
const glm::vec3 render::RenderEngine::DefaultBG     = glm::vec3(1.0f, 1.0f, 1.0f);


const glm::vec3 render::RenderEngine::CameraPos     = glm::vec3(0.0f, 1.0f, -2.0f);
const float render::RenderEngine::RotSpeed          = 40.0f;
const float render::RenderEngine::ScaleSpeed        = 1.0f;
const float render::RenderEngine::ZoomSpeed         = 5.0f;

const glm::vec2 render::RenderEngine::ZLim          = glm::vec2(0.01f, 100.0f);
const glm::vec2 render::RenderEngine::ZoomLim       = glm::vec2(1.0f/2.0f, 2.0f);
const glm::vec2 render::RenderEngine::PitchLim      = glm::radians(glm::vec2(-30.0f, 90.0f));
const glm::vec2 render::RenderEngine::ScaleLim      = glm::vec2(1e-2f, 1e2f);


float render::RenderEngine::Pitch                   = render::RenderEngine::DefaultPitch;
float render::RenderEngine::Yaw                     = render::RenderEngine::DefaultYaw;
float render::RenderEngine::Zoom                    = render::RenderEngine::DefaultZoom;
float render::RenderEngine::Scale                   = render::RenderEngine::DefaultScale;
glm::vec3 render::RenderEngine::BGCol               = render::RenderEngine::DefaultBG;


void render::RenderEngine::Init()
{
    static bool IsInit = false;
    if (IsInit)
        throw std::exception("OpenGL already initialized.");
    IsInit = true;

    if (!glfwInit())
        throw std::exception("Cannot initialize GLFW.");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWmonitor* Monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* VidMode = glfwGetVideoMode(Monitor);
    Window = glfwCreateWindow(VidMode->width, VidMode->height, "Slime on Manifolds", Monitor, NULL);
    if (Window == NULL)
        throw std::exception("Failed to create GLFW window.");
    glfwMakeContextCurrent(Window);
    glfwGetWindowSize(Window, &WinWidth, &WinHeight);

    int GLADSTatus = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    if (!GLADSTatus)
        throw std::exception("Failed to load GLAD.");

    glEnable(GL_DEPTH_TEST);
}


bool render::RenderEngine::ShouldClose()
{
    return glfwWindowShouldClose(Window);
}

void render::RenderEngine::Close()
{
    glfwSetWindowShouldClose(Window, true);
}


void render::RenderEngine::XRotation(float Degrees, float DeltaT)
{
    Pitch += glm::radians(Degrees) * DeltaT * RotSpeed;
    Pitch = glm::clamp(Pitch, PitchLim[0], PitchLim[1]);
}

void render::RenderEngine::YRotation(float Degrees, float DeltaT)
{
    Yaw += glm::radians(Degrees) * DeltaT * RotSpeed;
}

void render::RenderEngine::ChangeScale(float Factor, float DeltaT)
{
    Scale += Factor * DeltaT * ScaleSpeed;
    Scale = glm::clamp(Scale, ScaleLim[0], ScaleLim[1]);
}

void render::RenderEngine::ChangeZoom(float Factor, float DeltaT)
{
    Zoom += Factor * DeltaT * ZoomSpeed;
    Zoom = glm::clamp(Zoom, ZoomLim[0], ZoomLim[1]);
}

void render::RenderEngine::ChangeBackground(const glm::vec3& Color)
{
    BGCol = Color;
    BGCol = glm::clamp(BGCol, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
}


float render::RenderEngine::GetPitch() { return Pitch; }
float render::RenderEngine::GetYaw() { return Yaw; }
float render::RenderEngine::GetZoom() { return Zoom; }
float render::RenderEngine::GetScale() { return Scale; }
glm::vec3 render::RenderEngine::GetBackground() { return BGCol; }
glm::vec3 render::RenderEngine::GetCamera() { return CameraPos; }

glm::mat4 render::RenderEngine::GetModelMatrix()
{
    glm::mat4 Model = glm::identity<glm::mat4>();
    Model = glm::rotate(Model, Yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::vec3 Right(glm::cos(Yaw), 0.0f, glm::sin(Yaw));
    Model = glm::rotate(Model, Pitch, Right);
    Model = glm::scale(Model, glm::vec3(Scale, Scale, Scale));
    return Model;
}

glm::mat4 render::RenderEngine::GetViewMatrix()
{
    return glm::lookAt(CameraPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 render::RenderEngine::GetProjectionMatrix()
{
    float FoV = Zoom * glm::radians(DefaultFoV);
    float Aspect = (float)WinWidth / (float)WinHeight;
    return glm::perspective(FoV, Aspect, ZLim[0], ZLim[1]);
}


void render::RenderEngine::SetFramebufferSizeCallback(GLFWframebuffersizefun cb)
{
    glfwSetFramebufferSizeCallback(Window, cb);
}

void render::RenderEngine::SetScrollCallback(GLFWscrollfun cb)
{
    glfwSetScrollCallback(Window, cb);
}