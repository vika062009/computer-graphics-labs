#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec3 out_color;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;

    vec4 camera_position;
    vec4 light_direction;
    vec4 light_color_intensity;
    vec4 ambient_color_intensity;
} scene;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec4 tint;
    vec4 specular_shininess;
} object;

void main()
{
    vec4 world_position =
        object.model * vec4(in_position, 1.0);

    mat3 normal_matrix =
        transpose(inverse(mat3(object.model)));

    out_world_position = world_position.xyz;
    out_world_normal = normal_matrix * in_normal;
    out_color = in_color;

    gl_Position = scene.view_projection * world_position;
}