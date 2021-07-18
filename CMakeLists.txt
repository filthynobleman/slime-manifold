cmake_minimum_required(VERSION 3.19)
project(SlimeSimulation LANGUAGES C CXX CUDA)

# Search for GLFW and OpenGL
list(APPEND CMAKE_PREFIX_PATH "C:/include/GLFW")
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)

# Add include dependencies
include_directories("C:/include/GLFW/include")
include_directories(include)
include_directories(C:/include/GLAD)
include_directories(C:/include/GLM)
include_directories(C:/include/STB)

# Compile GLAD
add_library(GLAD STATIC C:/include/GLAD/src/glad.c)

# Compile STB
add_library(STB STATIC src/stb_image.cpp)

# Compile libraries
add_library(Mesh STATIC src/mesh/IO.cpp
                        src/mesh/Mesh.cpp)

add_library(Render STATIC   src/render/Material.cpp
                            src/render/GraphicShader.cpp
                            src/render/RenderEngine.cpp
                            src/render/MeshRenderEngine.cpp)
add_library(Slime STATIC    src/slime/SimulationParameters.cpp
                            src/slime/SlimeSim3D.cpp
                            src/slime/InitAgents3D.cu
                            src/slime/UpdatePos3D.cu
                            src/slime/DiffuseTrail3D.cu
                            src/slime/SlimeExporter.cpp
                            src/slime/SlimeExporter.cu)

target_link_libraries(Render Mesh STB)


# Executables
add_executable(Slime3D src/slime3d.cpp)
target_compile_options(Slime3D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

target_link_libraries(Slime3D Render Mesh Slime STB GLAD glfw OpenGL::GL cudart)


# Copy shaders to build directory
# UNCOMMENT FOR RELEASE
# list(APPEND Shaders "ScreenTexture.vert")
# list(APPEND Shaders "ScreenTexture.frag")
# list(APPEND Shaders "Slime2D.compute")
# foreach(s IN ZIP_LISTS Shaders)
#     file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${s}
#          DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shaders/${s})
# endforeach()