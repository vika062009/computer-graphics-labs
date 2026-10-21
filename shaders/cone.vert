#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec3 out_color;

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
} model_uniforms;

layout(push_constant) uniform DrawData {
    mat4 mvp;
    vec4 color;
} draw_data;

void main()
{
    if (draw_data.color.w > 0.5) {
        gl_Position = draw_data.mvp * vec4(in_position, 1.0);

        out_world_position = vec3(0.0);
        out_world_normal = vec3(0.0);
        out_color = draw_data.color.rgb;

        return;
    }

    vec4 world_position =
        model_uniforms.model * vec4(in_position, 1.0);

    mat3 normal_matrix =
        transpose(inverse(mat3(model_uniforms.model)));

    out_world_position = world_position.xyz;
    out_world_normal = normal_matrix * in_normal;
    out_color = in_color;

    gl_Position =
        scene_uniforms.view_projection * world_position;
}