/**
 * @file        GraphicShader.hpp
 * 
 * @brief       Class for a graphic shader.
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
#include <unordered_map>

#include <render/Material.hpp>


namespace render
{
    
class GraphicShader
{
private:
    std::string     VertFile;
    std::string     VertCode;
    std::string     FragFile;
    std::string     FragCode;
    GLuint          ProgID;

public:
    GraphicShader(const std::string& vs, const std::string& fs);
    ~GraphicShader();

    void Compile();
    void Use();

    void SetFloat(const std::string& Attrib, float Value);
    void SetVec2(const std::string& Attrib, const glm::vec2& Value);
    void SetVec3(const std::string& Attrib, const glm::vec3& Value);
    void SetVec4(const std::string& Attrib, const glm::vec4& Value);
    void SetMat2(const std::string& Attrib, const glm::mat2& Value);
    void SetMat3(const std::string& Attrib, const glm::mat3& Value);
    void SetMat4(const std::string& Attrib, const glm::mat4& Value);
    void SetLight(const std::string& Attrib, const Light& Value);
    void SetMaterial(const std::string& Attrib, const Material& Value);
    void SetTexture(const std::string& Attrib, const Texture& Value);

    
    static void CheckCompileErrors(GLuint Prog, GLenum Type, const std::string& filename);
};

} // namespace render
