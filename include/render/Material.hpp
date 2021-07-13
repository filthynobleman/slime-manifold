/**
 * @file        Material.hpp
 * 
 * @brief       Define materials and textures.
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

namespace render
{
    
struct Material
{
    glm::vec3       Ambient;
    glm::vec3       Diffuse;
    glm::vec3       Specular;
    float           SpecularExp;
};

struct Texture
{
    GLuint          BufIdx;
    GLuint          BindIdx;
    GLuint          Width;
    GLuint          Height;
    GLenum          Format;
    GLint           InFormat;
};

struct Light
{
    glm::vec3       Position;
    glm::vec3       Ambient;
    glm::vec3       Diffuse;
    glm::vec3       Specular;
};


Material    LoadMaterial(const std::string& filename);                          // only reads Ka, Kd, Ks, Ns from .mtl files
Light       LoadLight(const std::string& filename);                             // same as .mtl files, but with parameter XYZ and without Ns
Texture     LoadTexture(const std::string& filename);                           // uint texture
Texture     CreateTexture(GLuint Width, GLuint Height, GLenum Format);          // float texture



} // namespace render
