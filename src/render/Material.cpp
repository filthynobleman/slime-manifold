#include <render/Material.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>
#include <stb_image.h>

using namespace render;


Material render::LoadMaterial(const std::string& filename)
{
    Material Mat;
    std::ifstream Stream;
    Stream.open(filename, std::ios::in);
    if (!Stream.is_open())
    {
        std::stringstream ss;
        ss << "Cannot open file " << filename << " for reading";
        throw std::exception(ss.str().c_str());
    }

    std::string Line;
    while(!Stream.eof())
    {
        std::getline(Stream, Line);

        if (Line.rfind("Ka", 0) == 0)
        {
            std::sscanf(Line.c_str(), "Ka %f %f %f",
                        &(Mat.Ambient.x),
                        &(Mat.Ambient.y),
                        &(Mat.Ambient.z));
        }
        else if (Line.rfind("Kd", 0) == 0)
        {
            std::sscanf(Line.c_str(), "Kd %f %f %f",
                        &(Mat.Diffuse.x),
                        &(Mat.Diffuse.y),
                        &(Mat.Diffuse.z));
        }
        else if (Line.rfind("Ks", 0) == 0)
        {
            std::sscanf(Line.c_str(), "Ks %f %f %f",
                        &(Mat.Specular.x),
                        &(Mat.Specular.y),
                        &(Mat.Specular.z));
        }
        else if (Line.rfind("Ns", 0) == 0)
            std::sscanf(Line.c_str(), "Ns %f", &(Mat.SpecularExp));
    }

    Stream.close();

    return Mat;
}


Light render::LoadLight(const std::string& filename)
{
    Light L;
    std::ifstream Stream;
    Stream.open(filename, std::ios::in);
    if (!Stream.is_open())
    {
        std::stringstream ss;
        ss << "Cannot open file " << filename << " for reading";
        throw std::exception(ss.str().c_str());
    }

    std::string Line;
    while(!Stream.eof())
    {
        std::getline(Stream, Line);

        if (Line.rfind("Ka", 0) == 0)
        {
            std::sscanf(Line.c_str(), "Ka %f %f %f",
                        &(L.Ambient.x),
                        &(L.Ambient.y),
                        &(L.Ambient.z));
        }
        else if (Line.rfind("Kd", 0) == 0)
        {
            std::sscanf(Line.c_str(), "Kd %f %f %f",
                        &(L.Diffuse.x),
                        &(L.Diffuse.y),
                        &(L.Diffuse.z));
        }
        else if (Line.rfind("Ks", 0) == 0)
        {
            std::sscanf(Line.c_str(), "Ks %f %f %f",
                        &(L.Specular.x),
                        &(L.Specular.y),
                        &(L.Specular.z));
        }
        else if (Line.rfind("XYZ", 0) == 0)
        {
            std::sscanf(Line.c_str(), "XYZ %f %f %f",
                        &(L.Position.x),
                        &(L.Position.y),
                        &(L.Position.z));
        }
    }

    Stream.close();

    return L;
}

Texture render::LoadTexture(const std::string& filename, unsigned char** data)
{
    unsigned char* Buf;
    Texture Tex;

    int Width, Height, NumCh;
    Buf = stbi_load(filename.c_str(), &Width, &Height, &NumCh, 0);
    if (Buf == NULL)
    {
        std::stringstream ss;
        ss << "Cannot load image from file " << filename << ".";
        throw std::exception(ss.str().c_str());
    }
    Tex.Width = Width;
    Tex.Height = Height;

    switch (NumCh)
    {
    case 1:
        Tex.Format = GL_RED;
        break;
    case 2:
        Tex.Format = GL_RG;
        break;
    case 3:
        Tex.Format = GL_RGB;
        break;
    case 4:
        Tex.Format = GL_RGBA;
        break;
    
    case 0:
        stbi_image_free(Buf);
        throw std::exception("Cannot have textures with 0 channels.");
        break;

    default:
        stbi_image_free(Buf);
        std::stringstream ss;
        ss << "Texture with more than 4 channels are not supported. Given texture has " << NumCh << " channels.";
        throw std::exception(ss.str().c_str());
        break;
    }
    Tex.InFormat = Tex.Format;

    glGenTextures(1, &(Tex.BufIdx));
    glBindTexture(GL_TEXTURE_2D, Tex.BufIdx);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, Tex.InFormat, Tex.Width, Tex.Height, 0, Tex.Format, GL_UNSIGNED_BYTE, Buf);
    glGenerateMipmap(GL_TEXTURE_2D);

    if (data == NULL)
        stbi_image_free(Buf);
    else
        *data = Buf;

    return Tex;
}

Texture render::LoadTexture(const std::string& filename)
{
    return LoadTexture(filename, NULL);
}



Texture render::CreateTexture(GLuint Width, GLuint Height, GLenum Format)
{
    Texture Tex;
    Tex.Width = Width;
    Tex.Height = Height;
    Tex.Format = Format;
    switch (Tex.Format)
    {
    case GL_RED:
        Tex.InFormat = GL_R32F;
        break;
    case GL_RG:
        Tex.InFormat = GL_RG32F;
        break;
    case GL_RGB:
        Tex.InFormat = GL_RGB32F;
        break;
    case GL_RGBA:
        Tex.InFormat = GL_RGBA32F;
        break;
    
    default:
        std::stringstream ss;
        ss << "Format " << Format << " is not supported. Supported formats are:" << std::endl;
        ss << "    " << GL_RED << std::endl;
        ss << "    " << GL_RG << std::endl;
        ss << "    " << GL_RGB << std::endl;
        ss << "    " << GL_RGBA << std::endl;
        throw std::exception(ss.str().c_str());
    }

    glGenTextures(1, &(Tex.BufIdx));
    glBindTexture(GL_TEXTURE_2D, Tex.BufIdx);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, Tex.InFormat, Tex.Width, Tex.Height, 0, Tex.Format, GL_FLOAT, NULL);
    glGenerateMipmap(GL_TEXTURE_2D);

    return Tex;
}