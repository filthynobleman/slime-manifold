#include <slime/SlimeExporter.hpp>
#include <utils/CUDAUtils.h>

using namespace slime;


slime::SlimeExporter::SlimeExporter(const std::string& ExportDirectory, const render::Texture& Tex)
{
    ExpDir = ExportDirectory;
    NumFrames = 0;
    Width = Tex.Width;
    Height = Tex.Height;

    ExpTex = (unsigned char*)calloc(Width * Height, sizeof(unsigned char));
    if (ExpTex == NULL)
    {
        std::stringstream ss;
        ss << "Cannot allocate a " << Width << " x " << Height << " export texture.";
        throw std::exception(ss.str().c_str());
    }
    cudaErrorCheck(cudaCalloc<unsigned char>(&dExpTex, Width * Height));
}


slime::SlimeExporter::~SlimeExporter()
{
    cudaErrorCheck(cudaFree(dExpTex));
}