#version 450 core

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec3 in_color;
layout(location = 0) out vec4 out_color;

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

struct PointLight {
    vec3 position;
    vec3 color;
};

struct SpotLight {
    vec3 position;
    vec3 color;
    vec3 direction;
    float angle;      
    float outer_angle; 
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

vec3 safe_normalize(vec3 value)
{
    float length_squared = dot(value, value);
    if (length_squared < 0.00000001) {
        return vec3(0.0);
    }
    return value * inversesqrt(length_squared);
}

vec3 evaluate_light(vec3 N, vec3 V, vec3 L, vec3 albedo, vec3 radiance)
{
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) {
        return vec3(0.0);
    }

    vec3 H = safe_normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float alpha = max(model_uniforms.alpha, 1.0);

    vec3 diffuse = albedo * NdotL;
    vec3 specular = model_uniforms.specular * pow(NdotH, alpha);

    return (diffuse + specular) * NdotL * radiance;
}

void main()
{
    vec3 N = safe_normalize(in_world_normal);
    vec3 V = safe_normalize(scene_uniforms.camera_position - in_world_position);
    vec3 albedo = in_color * model_uniforms.albedo;
    vec3 result = albedo * scene_uniforms.ambient_light;

    vec3 L = safe_normalize(-scene_uniforms.directional_light_direction);
    result += evaluate_light(N, V, L, albedo, scene_uniforms.directional_light_color);

    uint point_count = min(scene_uniforms.point_light_count, uint(point_lights.length()));
    for (uint i = 0u; i < point_count; ++i) {
        PointLight light = point_lights[i];
        vec3 to_light = light.position - in_world_position;
        float distance_squared = dot(to_light, to_light);
        vec3 point_L = safe_normalize(to_light);
        vec3 radiance = light.color / max(distance_squared, 0.0001);
        result += evaluate_light(N, V, point_L, albedo, radiance);
    }

    uint spot_count = min(scene_uniforms.spot_light_count, uint(spot_lights.length()));
    for (uint i = 0u; i < spot_count; ++i) {
        SpotLight light = spot_lights[i];
        if (dot(light.direction, light.direction) < 0.00000001) {
            continue;
        }

        vec3 to_light = light.position - in_world_position;
        float distance_squared = dot(to_light, to_light);
        vec3 spot_L = safe_normalize(to_light);
        float cos_angle = dot(safe_normalize(light.direction), -spot_L);

        float cone_factor;
        if (light.angle > light.outer_angle) {
            cone_factor = smoothstep(light.outer_angle, light.angle, cos_angle);
        } else {
            cone_factor = step(light.outer_angle, cos_angle);
        }

        vec3 radiance = light.color * cone_factor / max(distance_squared, 0.0001);
        result += evaluate_light(N, V, spot_L, albedo, radiance);
    }

    out_color = vec4(result, 1.0);
}
