#version 450 core

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec4 out_color;

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

struct PointLight {
    vec4 position;
    vec4 color_intensity;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color_intensity;
    vec4 cone_angles;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    uvec4 header;
    PointLight lights[];
} point_lights;

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    uvec4 header;
    SpotLight lights[];
} spot_lights;

vec3 safe_normalize(vec3 value)
{
    float length_squared = dot(value, value);

    if (length_squared < 0.00000001) {
        return vec3(0.0);
    }

    return value * inversesqrt(length_squared);
}

vec3 evaluate_light(
    vec3 N,
    vec3 V,
    vec3 L,
    vec3 albedo,
    vec3 radiance)
{
    float NdotL = max(dot(N, L), 0.0);

    if (NdotL <= 0.0) {
        return vec3(0.0);
    }

    vec3 H = safe_normalize(L + V);

    float NdotH = max(dot(N, H), 0.0);
    float shininess = max(object.specular_shininess.w, 1.0);

    vec3 diffuse = albedo * NdotL;

    vec3 specular =
        object.specular_shininess.rgb *
        pow(NdotH, shininess);

    return (diffuse + specular) * NdotL * radiance;
}

void main()
{
    vec3 N = safe_normalize(in_world_normal);

    vec3 V = safe_normalize(
        scene.camera_position.xyz - in_world_position
    );

    vec3 albedo = in_color * object.tint.rgb;

    vec3 result =
        albedo *
        scene.ambient_color_intensity.rgb *
        scene.ambient_color_intensity.w;

    vec3 directional_L =
        safe_normalize(-scene.light_direction.xyz);

    vec3 directional_radiance =
        scene.light_color_intensity.rgb *
        scene.light_color_intensity.w;

    result += evaluate_light(
        N,
        V,
        directional_L,
        albedo,
        directional_radiance
    );

    uint point_count = min(
        point_lights.header.x,
        uint(point_lights.lights.length())
    );

    for (uint i = 0u; i < point_count; ++i) {
        PointLight light = point_lights.lights[i];

        vec3 to_light =
            light.position.xyz - in_world_position;

        float distance_squared = dot(to_light, to_light);
        vec3 L = safe_normalize(to_light);

        float attenuation =
            1.0 / max(distance_squared, 0.0001);

        vec3 radiance =
            light.color_intensity.rgb *
            light.color_intensity.w *
            attenuation;

        result += evaluate_light(N, V, L, albedo, radiance);
    }

    uint spot_count = min(
        spot_lights.header.x,
        uint(spot_lights.lights.length())
    );

    for (uint i = 0u; i < spot_count; ++i) {
        SpotLight light = spot_lights.lights[i];

        if (dot(light.direction.xyz, light.direction.xyz)
            < 0.00000001) {
            continue;
        }

        vec3 to_light =
            light.position.xyz - in_world_position;

        float distance_squared = dot(to_light, to_light);
        vec3 L = safe_normalize(to_light);

        vec3 ray_direction =
            safe_normalize(light.direction.xyz);

        float cos_angle = dot(ray_direction, -L);

        float inner_cos = light.cone_angles.x;
        float outer_cos = light.cone_angles.y;

        float cone_factor;

        if (inner_cos > outer_cos) {
            cone_factor = smoothstep(
                outer_cos,
                inner_cos,
                cos_angle
            );
        }
        else {
            cone_factor = step(outer_cos, cos_angle);
        }

        float attenuation =
            1.0 / max(distance_squared, 0.0001);

        vec3 radiance =
            light.color_intensity.rgb *
            light.color_intensity.w *
            attenuation *
            cone_factor;

        result += evaluate_light(N, V, L, albedo, radiance);
    }

    out_color = vec4(result, 1.0);
}