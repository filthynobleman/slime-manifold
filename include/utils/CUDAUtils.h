#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <exception>

#include <glad/glad.h>

#include <cuda_gl_interop.h>



// Hash function www.cs.ubc.ca/~rbridson/docs/schechter-sca08-turbulence.pdf
__device__ inline unsigned int RandHash(unsigned int State)
{
    State ^= 2747636419u;
    State *= 2654435769u;
    State ^= State >> 16;
    State *= 2654435769u;
    State ^= State >> 16;
    State *= 2654435769u;
    return State;
}


__device__ inline float ScaleTo01(unsigned int State)
{
    return State / 4294967295.0f;
}

void RandomNoise(float* dImage, int Width, int Height);
// void RandomNoise(cudaArray_t dImage, int Width, int Height);



inline void CUDAErrorPrint(cudaError_t err, const char* code, const char* file, int line)
{
    std::stringstream ss;
    ss << "CUDA raised an error at " << file << ':' << line << std::endl;
    ss << code << std::endl;
    ss << "Error " << err << ": " << cudaGetErrorString(err) << std::endl;
    throw std::exception(ss.str().c_str());
}

template<typename T>
cudaError_t cudaCalloc(T** devPtr, size_t count, size_t size = sizeof(T))
{
    cudaError_t err;
    err = cudaMalloc<T>(devPtr, count * size);
    if (err != cudaSuccess)
        return err;
    err = cudaMemset(*devPtr, 0, count * size);
    return err;
}

template<typename T>
cudaError_t cudaAllocCopy(T** destPtr, const T* srcPtr, size_t count, size_t size = sizeof(T))
{
    cudaError_t err;
    err = cudaMalloc<T>(destPtr, count * size);
    if (err != cudaSuccess)
        return err;
    err = cudaMemcpy(*destPtr, srcPtr, count * size, cudaMemcpyHostToDevice);
    return err;
}



#define __STR(x) #x
#define STR(x) __STR(x)

#define cudaErrorCheck(err)     do {\
                                    if (err != cudaSuccess)\
                                        CUDAErrorPrint(err, STR(err), __FILE__, __LINE__);\
                                } while (0)




inline void RandNoise(unsigned int TBO, int Width, int Height)
{
    // glBindTexture(GL_TEXTURE_2D, 0);
    // cudaGraphicsResource *cudaTex;
    // cudaErrorCheck(cudaGraphicsGLRegisterImage(&cudaTex, TBO, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    // cudaErrorCheck(cudaGraphicsMapResources(1, &cudaTex));
    // cudaArray *cudaTexArray;
    // cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&cudaTexArray, cudaTex, 0, 0));
    // RandomNoise(cudaTexArray, Width, Height);
    // cudaErrorCheck(cudaStreamSynchronize(0));
    // cudaErrorCheck(cudaGraphicsUnmapResources(1, &cudaTex));
    // cudaErrorCheck(cudaGraphicsUnregisterResource(cudaTex));
    // cudaErrorCheck(cudaStreamSynchronize(0));


    GLuint PBO;
    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, Width * Height * sizeof(float), NULL, GL_DYNAMIC_COPY);
    cudaErrorCheck(cudaGLRegisterBufferObject(PBO));

    float* dImage;
    cudaErrorCheck(cudaGLMapBufferObject((void**)&dImage, PBO));
    RandomNoise(dImage, Width, Height);
    cudaErrorCheck(cudaGLUnmapBufferObject(PBO));
    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBindTexture(GL_TEXTURE_2D, TBO);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Width, Height, GL_RED, GL_FLOAT, NULL);
}