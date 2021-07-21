/**
 * @file        SlimeExporter.hpp
 * 
 * @brief       Class handling the operations needed for exporting the slime texture
 * 
 * @details     
 * 
 * @author      Filippo Maggioli (maggioli@di.uniroma1.it)
 *              Sapienza, University of Rome - Department of Computer Science
 * 
 * @date 2021-07-15
 * 
 */
#pragma once

#include <string>
#include <render/Material.hpp>

namespace slime
{
    
class SlimeExporter
{
private:
    std::string     ExpDir;
    unsigned int    NumFrames;
    unsigned char*  ExpTex;
    unsigned char*  dExpTex;
    unsigned int    Width;
    unsigned int    Height;
    unsigned int    NumCh;
    
public:
    SlimeExporter(const std::string& ExportDirectory, const render::Texture& Tex);
    ~SlimeExporter();

    void ExportFrame(const float* dTexture);
};

} // namespace slime
