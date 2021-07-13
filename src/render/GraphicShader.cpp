#include <render/GraphicShader.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>

#include <glm/gtc/type_ptr.hpp>

using namespace render;


render::GraphicShader::GraphicShader(const std::string& vs, const std::string& fs)
{
    // Uninitialized program
    ProgID = 0;

    // Save shaders filenames
    VertFile = vs;
    FragFile = fs;

    // Read vertex shader
    std::ifstream Stream;
    Stream.open(vs, std::ios::in);
    if (!Stream.is_open())
    {
        std::stringstream ss;
        ss << "Cannot open file " << vs << " for reading.";
        throw std::exception(ss.str().c_str());
    }

    std::stringstream VCodeStream;
    VCodeStream << Stream.rdbuf();

    Stream.close();
    VertCode = VCodeStream.str();

    // Read fragment shader
    Stream.open(fs, std::ios::in);
    if (!Stream.is_open())
    {
        std::stringstream ss;
        ss << "Cannot open file " << fs << " for reading.";
        throw std::exception(ss.str().c_str());
    }

    std::stringstream FCodeStream;
    FCodeStream << Stream.rdbuf();

    Stream.close();
    FragCode = FCodeStream.str();
}

render::GraphicShader::~GraphicShader()
{
    if (ProgID != 0)
        glDeleteProgram(ProgID);
}


void render::GraphicShader::Compile()
{
    // Compile vertex shader
    GLuint VertID = glCreateShader(GL_VERTEX_SHADER);
    const char* VertSource = VertCode.c_str();
    glShaderSource(VertID, 1, &VertSource, NULL);
    glCompileShader(VertID);
    CheckCompileErrors(VertID, GL_VERTEX_SHADER, VertFile);

    // Compile fragment shader
    GLuint FragID = glCreateShader(GL_FRAGMENT_SHADER);
    const char* FragSource = FragCode.c_str();
    glShaderSource(FragID, 1, &FragSource, NULL);
    glCompileShader(FragID);
    CheckCompileErrors(FragID, GL_FRAGMENT_SHADER, FragFile);

    // Link
    ProgID = glCreateProgram();
    glAttachShader(ProgID, VertID);
    glAttachShader(ProgID, FragID);
    glLinkProgram(ProgID);
    CheckCompileErrors(ProgID, GL_PROGRAM, "");

    // Delete shaders
    glDeleteShader(VertID);
    glDeleteShader(FragID);
}

void render::GraphicShader::Use()
{
    glUseProgram(ProgID);
}


void render::GraphicShader::SetFloat(const std::string& Attrib, float Value)
{
    GLint Loc = glGetUniformLocation(ProgID, Attrib.c_str());
    if (Loc < 0)
    {
        std::stringstream ss;
        ss << "Cannot find uniform location with name " << Attrib << '.';
        throw std::exception(ss.str().c_str());
    }
    glUniform1f(Loc, Value);
}

void render::GraphicShader::SetVec2(const std::string& Attrib, const glm::vec2& Value)
{
    GLint Loc = glGetUniformLocation(ProgID, Attrib.c_str());
    if (Loc < 0)
    {
        std::stringstream ss;
        ss << "Cannot find uniform location with name " << Attrib << '.';
        throw std::exception(ss.str().c_str());
    }
    glUniform2fv(Loc, 1, glm::value_ptr(Value));
}

void render::GraphicShader::SetVec3(const std::string& Attrib, const glm::vec3& Value)
{
    GLint Loc = glGetUniformLocation(ProgID, Attrib.c_str());
    if (Loc < 0)
    {
        std::stringstream ss;
        ss << "Cannot find uniform location with name " << Attrib << '.';
        throw std::exception(ss.str().c_str());
    }
    glUniform3fv(Loc, 1, glm::value_ptr(Value));
}

void render::GraphicShader::SetVec4(const std::string& Attrib, const glm::vec4& Value)
{
    GLint Loc = glGetUniformLocation(ProgID, Attrib.c_str());
    if (Loc < 0)
    {
        std::stringstream ss;
        ss << "Cannot find uniform location with name " << Attrib << '.';
        throw std::exception(ss.str().c_str());
    }
    glUniform4fv(Loc, 1, glm::value_ptr(Value));
}

void render::GraphicShader::SetMat2(const std::string& Attrib, const glm::mat2& Value)
{
    GLint Loc = glGetUniformLocation(ProgID, Attrib.c_str());
    if (Loc < 0)
    {
        std::stringstream ss;
        ss << "Cannot find uniform location with name " << Attrib << '.';
        throw std::exception(ss.str().c_str());
    }
    glUniformMatrix2fv(Loc, 1, GL_FALSE, glm::value_ptr(Value));
}

void render::GraphicShader::SetMat3(const std::string& Attrib, const glm::mat3& Value)
{
    GLint Loc = glGetUniformLocation(ProgID, Attrib.c_str());
    if (Loc < 0)
    {
        std::stringstream ss;
        ss << "Cannot find uniform location with name " << Attrib << '.';
        throw std::exception(ss.str().c_str());
    }
    glUniformMatrix3fv(Loc, 1, GL_FALSE, glm::value_ptr(Value));
}

void render::GraphicShader::SetMat4(const std::string& Attrib, const glm::mat4& Value)
{
    GLint Loc = glGetUniformLocation(ProgID, Attrib.c_str());
    if (Loc < 0)
    {
        std::stringstream ss;
        ss << "Cannot find uniform location with name " << Attrib << '.';
        throw std::exception(ss.str().c_str());
    }
    glUniformMatrix4fv(Loc, 1, GL_FALSE, glm::value_ptr(Value));
}

void render::GraphicShader::SetLight(const std::string& Attrib, const Light& Value)
{
    SetVec3(Attrib + ".Position", Value.Position);
    SetVec3(Attrib + ".Ambient", Value.Ambient);
    SetVec3(Attrib + ".Diffuse", Value.Ambient);
    SetVec3(Attrib + ".Specular", Value.Ambient);
}

void render::GraphicShader::SetMaterial(const std::string& Attrib, const Material& Value)
{
    SetVec3(Attrib + ".Ambient", Value.Ambient);
    SetVec3(Attrib + ".Diffuse", Value.Ambient);
    SetVec3(Attrib + ".Specular", Value.Ambient);
    SetFloat(Attrib + ".SpecularExp", Value.SpecularExp);
}

void render::GraphicShader::SetTexture(const std::string& Attrib, const Texture& Value)
{
    GLint Loc = glGetUniformLocation(ProgID, Attrib.c_str());
    if (Loc < 0)
    {
        std::stringstream ss;
        ss << "Cannot find uniform location with name " << Attrib << '.';
        throw std::exception(ss.str().c_str());
    }
    glUniform1i(Loc, Value.BindIdx);
}


void render::GraphicShader::CheckCompileErrors(GLuint Prog, GLenum Type, const std::string& filename)
{
    GLint Success;
    GLchar InfoLog[4096];

    glGetShaderiv(Prog, GL_COMPILE_STATUS, &Success);
    if(!Success)
    {
        std::string TypeStr;
        switch (Type)
        {
        case GL_VERTEX_SHADER:
            TypeStr = "VERTEX_SHADER_COMPILATION_ERROR";
            break;
        
        case GL_FRAGMENT_SHADER:
            TypeStr = "FRAGMENT_SHADER_COMPILATION_ERROR";
            break;
        
        case GL_PROGRAM:
            TypeStr = "SHADER_PROGRAM_LINK_ERROR";
            break;

        case GL_COMPUTE_SHADER:
            TypeStr = "COMPUTE_SHADER_COMPILATION_ERROR";
            break;
        
        default:
            throw std::exception("Unsupported shader type.");
        }

        glGetShaderInfoLog(Prog, 1024, NULL, InfoLog);
        std::cerr << "ERROR::" << TypeStr << " on " << filename
                  << std::endl << InfoLog
                  << "\n -- --------------------------------------------------- -- " 
                  << std::endl;
    }
}