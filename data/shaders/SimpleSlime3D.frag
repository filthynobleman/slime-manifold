#version 440 core


out vec4 FragColor;


in vec3 FragPos;
in vec3 FragNorm;
in vec2 FragTex;

struct Material_t
{
    vec3   Ambient;
    vec3   Diffuse;
    vec3   Specular;
    float  SpecularExp;
};

struct Light_t
{
    vec3   Position;
    vec3   Ambient;
    vec3   Diffuse;
    vec3   Specular;
};

uniform vec3        CameraPos;
uniform Material_t  Material;
uniform Light_t     Light;
uniform sampler2D   OutTex;


void main()
{
    vec3 Ambient = Material.Ambient * Light.Ambient;

    vec3 Norm = normalize(FragNorm);
    vec3 LightDir = normalize(Light.Position - FragPos);
    float Diff = max(dot(LightDir, Norm), 0.0f);
    vec3 Diffuse = Light.Diffuse * (Diff * Material.Diffuse);

    vec3 ViewDir = normalize(CameraPos - FragPos);
    vec3 ReflectionDir = reflect(-LightDir, Norm);
    float Spec = pow(max(dot(ViewDir, ReflectionDir), 0.0f), Material.SpecularExp);
    vec3 Specular = Light.Specular * (Spec * Material.Specular);

    vec4 TexCol = texture(OutTex, FragTex);
    vec3 Result = Ambient + Diffuse + Specular;
    FragColor = vec4(Result, 1.0f) * (1 - TexCol.x);
}