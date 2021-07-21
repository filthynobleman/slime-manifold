#include <slime/SlimeExporter.hpp>
#include <utils/CUDAUtils.h>

using namespace slime;


slime::SlimeExporter::SlimeExporter(const std::string& ExportDirectory, const render::Texture& Tex)
{
    ExpDir = ExportDirectory;
    NumFrames = 0;
    Width = Tex.Width;
    Height = Tex.Height;

    switch (Tex.Format)
    {
    case GL_RED:
        NumCh = 1;
        break;
    case GL_RG:
        NumCh = 2;
        break;
    case GL_RGB:
        NumCh = 3;
        break;

    default:
        throw std::exception("Valid export formats are GL_RED, GL_RG and GL_RGB.");
    }

    ExpTex = (unsigned char*)calloc(Width * Height * NumCh, sizeof(unsigned char));
    if (ExpTex == NULL)
    {
        std::stringstream ss;
        ss << "Cannot allocate a " << Width << " x " << Height << " export texture with " << NumCh << " channels.";
        throw std::exception(ss.str().c_str());
    }
    cudaErrorCheck(cudaCalloc<unsigned char>(&dExpTex, Width * Height * NumCh));
}


slime::SlimeExporter::~SlimeExporter()
{
    cudaErrorCheck(cudaFree(dExpTex));
}