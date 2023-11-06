# Slime Mold Simulation on Manifold Meshes

This repository contains the code for computing a slime mold simulation on triangular manifold meshes, as described in the manuscript *MoMaS: Mold Manifold Simulation for real-time procedural texturing*.  
- DOI: https://doi.org/10.1111/cgf.14697
- PDF: https://diglib.eg.org/handle/10.1111/cgf14697  

> **Warning**
> The code contained in this repository is only intended for the reproducibility of the manuscript. It is not intended for easy readability or cross-platform compilation.  


## Building
This section briefly describes the steps required to build the application.  
The build has been tested on Windows, but it **should** work on any operating system.

### Requirements
The building process requires a working C++ compiler and relies on the following third party tools:
- CUDA (tested on v10.2 and v11.6)
- CMake (tested on v3.19 and v3.23)  

Additionally, the following external libraries are provided with this repository:
- GLAD
- GLFW
- GLM
- STB Image

### Preparation
After the repository has been cloned, update the submodules with the command
```sh
git submodule update --init --recursive --remote
```

Before building the application, GLFW must be built and installed into a proper directory. From the project's root directory, execute the following commands:
```sh
mkdir build-glfw
cd build-glfw
cmake .. -DCMAKE_INSTALL_PREFIX="../install"
cmake --build . --config release
cmake --install .
```

### Compilation
The compilation is completely handled with CMake. From the project's root directory, execute the following commands:
```sh
mkdir build
cd build
cmake ..
cmake --build . --config release
```


## Execution
The compilation process will generate an executable `Slime3D.exe` in the directory `Release\` inside the build directory (this is true in a Windows environment, on Linux the executables are generated directly inside the build directory).  
The syntax to execute the application is the following
```sh
Slime3D.exe config_file obj_file [ tex_res [export_video [mtl_file [light_file]]]]
```
where:
- `config_file` is a configuration file for the simulation (examples in `data/configs`);
- `obj_file` is a triangular mesh in OBJ format, which **_must_** also define UV coordinates and normals at each face;
- `tex_res` is the texture resolution (by default `1024`);
- `export_video` is either `1` or `0` and determines if the textures must be exported to file (default is `0`);
- `mtl_file` is the material file for rendering (default is `../data/meshes/default.mtl`);
- `light_file` is the file defining the environmental light (default is `../data/meshes/default.light`).
