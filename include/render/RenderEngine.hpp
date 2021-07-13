/**
 * @file        RenderEngine.hpp
 * 
 * @brief       Static class for render engine.
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

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <string>

#include <mesh/Mesh.hpp>


namespace render
{
    
class RenderEngine
{
protected:
    // Window properties
    static GLFWwindow*          Window;
    static int                  WinWidth;
    static int                  WinHeight;

    // Attributes bounds
    static const glm::vec2      ZLim;
    static const glm::vec2      PitchLim;
    static const glm::vec2      ScaleLim;
    static const glm::vec2      ZoomLim;

    // Transform attributes
    static float                Pitch;
    static float                Yaw;
    static float                Zoom;
    static float                Scale;

    // Scene properties
    static const glm::vec3      CameraPos;
    static const float          RotSpeed;
    static const float          ScaleSpeed;
    static const float          ZoomSpeed;
    static glm::vec3            BGCol;


    // Default values
    static const float          DefaultPitch;
    static const float          DefaultYaw;
    static const float          DefaultFoV;
    static const float          DefaultZoom;
    static const float          DefaultScale;
    static const glm::vec3      DefaultBG;

private:
    // Constructor and destructor must be inaccessible
    RenderEngine() {};
    ~RenderEngine() {};



public:
    // Initialize OpenGL
    static void Init();

    // Window handlers
    static bool ShouldClose();
    static void Close();

    // Override these
    static void Draw() {};
    static void ProcessInput() {};

    // Setters
    static void XRotation(float Degrees, float DeltaT = 1.0f);
    static void YRotation(float Degrees, float DeltaT = 1.0f);
    static void ChangeScale(float Factor, float DeltaT = 1.0f);
    static void ChangeZoom(float Factor, float DeltaT = 1.0f);
    static void ChangeBackground(const glm::vec3& Color);
    
    // Getters
    static float GetPitch();
    static float GetYaw();
    static float GetScale();
    static float GetZoom();
    static glm::vec3 GetBackground();
    static glm::vec3 GetCamera();
    static glm::mat4 GetModelMatrix();
    static glm::mat4 GetViewMatrix();
    static glm::mat4 GetProjectionMatrix();

    // Window callbacks
    static void SetFramebufferSizeCallback(GLFWframebuffersizefun cb);
    static void SetScrollCallback(GLFWscrollfun cb);

};

} // namespace render
