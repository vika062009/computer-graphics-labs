#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec3 out_uv;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    vec3 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec3 albedo;
    vec3 specular;
    float alpha;
    uint unlit;
} model_uniforms;

void main()
{
    vec4 world_position = model_uniforms.model * vec4(in_position, 1.0);

    mat3 normal_matrix = model_uniforms.unlit != 0u
        ? mat3(1.0)
        : transpose(inverse(mat3(model_uniforms.model)));

    out_world_position = world_position.xyz;
    out_world_normal = normal_matrix * in_normal;
    float uv_weight = (in_position.y < -0.999) ? 0.0 : 1.0;
    out_uv = vec3(in_uv.x * uv_weight, in_uv.y, uv_weight);

    gl_Position = scene_uniforms.view_projection * world_position;
}