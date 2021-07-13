/**
 * @file        TextureRenderEngine.hpp
 * 
 * @brief       Class for rendering a texture on screen
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


namespace render
{
    
class TextureRenderEngine : public RenderEngine
{
public:
    static void Draw();
    static void ProcessInput();
};


} // namespace render
