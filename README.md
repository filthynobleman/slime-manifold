# Slime Mold Simulation on Manifold Meshes

This repository contains the code for computing a slime mold simulation on triangular manifold meshes, as described in the manuscript *MoMaS: Mold Manifold Simulation for real-time procedural texturing* (a link will be added soon).  

> **Warning**
> The code contained in this repository is only intended for the reproducibility of the manuscript. It is not intended for easy readability or cross-platform compilation.  
> _**The repository will be soon marked as deprecated and replaced with a definitive version.**_


## Installation

This section briefly describes the steps required to run the program.

### Requirements
In order to compile and execute the simulation, the following libraries and programs are needed:
 - `GLAD`
 - `GLFW`
 - `GLM`
 - `STB Image`
 - `CUDA` (tested on v10.2)
 - `CMake` (tested on v3.19)

The build has been tested on Windows, but it **should** work on any operating system.


### Compilation
To compile the source code please execute the following commands:
```
    mkdir build
    cmake ..
    cmake --build .
```


## Execution
The compilation process will generate an executable `Slime3D.exe` in the directory `Debug\` inside the build directory.  
The program can then be executed by typing
```
    .\Debug\Slime3D.exe
```
from the build directory.
