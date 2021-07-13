/**
 * @file        MeshRenderEngine.hpp
 * 
 * @brief       Extends RenderEngine to render meshes.
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

#include <render/RenderEngine.hpp>
#include <render/GraphicShader.hpp>
#include <render/Material.hpp>
#include <vector>

namespace render
{
    
class MeshRenderEngine : public RenderEngine
{
private:
    static GLuint VAO;
    static GLuint VBO;
    static GLuint EBO;
    static GLuint NTris;

    static GraphicShader*               Shader;
    static Material                     MeshMaterial;
    static Light                        SceneLight;
    static std::vector<Texture>         Textures;
    static std::vector<std::string>     TexNames;

    static double   MouseOldX;
    static double   MouseOldY;
    static float    CurTime;
    static float    OldTime;
    static float    LastActionTime;
    static float    ActionDelay;

    static GLenum   PolyModes[2];
    static int      CurMode;
    
public:
    static void CreateBuffers(const mesh::Mesh& M);
    static void SetLight(const std::string& filename);
    static void SetLight(const Light& L);
    static void SetMaterial(const std::string& filename);
    static void SetMaterial(const Material& M);
    static void AddTexture(const std::string& TexName, const Texture& Tex);
    static void CreateShader(const std::string& vert, const std::string& frag);
    static void SetCallbacks();

    static float GetDeltaTime();

    static void BindAllTextures();
    static void Draw();
    static void ProcessInput();
};


} // namespace render
