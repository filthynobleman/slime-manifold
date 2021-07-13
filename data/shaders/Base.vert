#version 440 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNorm;
layout(location = 2) in vec2 aTex;

uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;
uniform mat3 ModelInv;

out vec3 FragPos;
out vec3 FragNorm;
out vec2 FragTex;


void main()
{
    vec4 ModelPos = Model * vec4(aPos, 1.0f);
    gl_Position = Projection * View * ModelPos;
    FragPos = vec3(ModelPos);
    FragNorm = ModelInv * aNorm;
    FragTex = aTex;
}