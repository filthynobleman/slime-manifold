cmake_minimum_required(VERSION 3.19)
project(SlimeSimulation LANGUAGES C CXX CUDA)

# Search for GLFW and OpenGL
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/install")
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)

# Add include dependencies
include_directories("${CMAKE_SOURCE_DIR}/ext/glfw/include")
include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/ext/glad/include")
include_directories("${CMAKE_SOURCE_DIR}/ext/glm")
include_directories("${CMAKE_SOURCE_DIR}/ext/stb")

# Compile GLAD
add_library(GLAD STATIC "${CMAKE_SOURCE_DIR}/ext/glad/src/glad.c")

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
add_executable(SlimeObstacle3D src/slime3dobstacle.cpp)
add_executable(SlimeInitMap3D src/slime3dinitmap.cpp)
target_compile_options(Slime3D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
target_compile_options(SlimeObstacle3D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --generate-line-info
                        --use_fast_math
                        --relocatable-device-code=true
                        >)

target_link_libraries(Slime3D Render Mesh Slime STB GLAD glfw OpenGL::GL cudart)
target_link_libraries(SlimeObstacle3D Render Mesh Slime STB GLAD glfw OpenGL::GL cudart)
target_link_libraries(SlimeInitMap3D Render Mesh Slime STB GLAD glfw OpenGL::GL cudart)


# Copy shaders to build directory
# UNCOMMENT FOR RELEASE
# list(APPEND Shaders "ScreenTexture.vert")
# list(APPEND Shaders "ScreenTexture.frag")
# list(APPEND Shaders "Slime2D.compute")
# foreach(s IN ZIP_LISTS Shaders)
#     file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${s}
#          DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shaders/${s})
# endforeach()