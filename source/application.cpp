#include "application.hpp"

#include <imgui.h>
#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include <vk_mem_alloc.h>

namespace application {

    namespace {

        constexpr uint32_t CONE_COUNT = 3;

        constexpr uint32_t CONE_SEGMENTS = 50;
        constexpr float CONE_RADIUS = 0.5f;
        constexpr float CONE_HEIGHT = 1.0f;
        constexpr float PI = 3.14159265358979323846f;

        constexpr const char* VERT_SHADER_PATH = "../shaders/cone.vert.spv";
        constexpr const char* FRAG_SHADER_PATH = "../shaders/cone.frag.spv";

        struct Vertex {
            float position[3];
            float color[3];
            float normal[3];
        };

        struct SceneUniforms {
            float view_projection[4][4];

            float camera_position[4];
            float light_direction[4];
            float light_color_intensity[4];
            float ambient_color_intensity[4];
        };

        struct ModelUniforms {
            float model[4][4];
            float tint[4];
            float specular_shininess[4];
        };

        static_assert(sizeof(SceneUniforms) == 128);
        static_assert(sizeof(ModelUniforms) == 96);

        struct PointLight {
            float position[4];
            float color_intensity[4];
        };

        struct SpotLight {
            float position[4];
            float direction[4];
            float color_intensity[4];
            float cone_angles[4];
        };

        struct LightBufferHeader {
            uint32_t count;
            uint32_t padding[3];
        };

        static_assert(sizeof(PointLight) == 32);
        static_assert(sizeof(SpotLight) == 64);
        static_assert(sizeof(LightBufferHeader) == 16);

        std::vector<PointLight> g_point_lights;
        std::vector<SpotLight> g_spot_lights;

        struct LightStorageBuffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = nullptr;
            void* memory = nullptr;
            VkDeviceSize size = 0;
        };

        LightStorageBuffer g_point_light_buffer;
        LightStorageBuffer g_spot_light_buffer;

        bool g_lights_dirty = false;

        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;
        Vertex* vk_vertex_buffer_memory = nullptr;

        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;
        uint32_t* vk_index_buffer_memory = nullptr;
        uint32_t vk_index_count = 0;

        VkBuffer vk_scene_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_scene_uniform_allocation = nullptr;
        SceneUniforms* vk_scene_uniform_memory = nullptr;

        VkBuffer vk_uniform_buffer[CONE_COUNT] = {};
        VmaAllocation vk_uniform_buffer_allocation[CONE_COUNT] = {};
        ModelUniforms* vk_uniform_buffer_memory[CONE_COUNT] = {};

        VkDescriptorSetLayout vk_scene_set_layout = VK_NULL_HANDLE;
        VkDescriptorSet vk_scene_descriptor_set = VK_NULL_HANDLE;

        VkDescriptorSetLayout vk_model_set_layout = VK_NULL_HANDLE;
        VkDescriptorSet vk_descriptor_set[CONE_COUNT] = {};

        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;

        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline vk_pipeline = VK_NULL_HANDLE;

        VkShaderModule vk_vertex_shader = VK_NULL_HANDLE;
        VkShaderModule vk_fragment_shader = VK_NULL_HANDLE;

        float g_view[4][4];

        enum class ProjectionMode { Perspective, Orthographic };

        struct CameraState {
            float position[3] = { 0.0f, 0.0f, -3.0f };
            float rotation_degrees[3] = { 0.0f, 0.0f, 0.0f };

            ProjectionMode projection_mode = ProjectionMode::Perspective;
            float fov_degrees = 60.0f;
            float ortho_scale = 1.5f;

            float projection[4][4] = {};
        };

        CameraState g_camera;

        float g_light_direction[3] = { -0.5f, 1.0f, 1.0f };
        float g_light_color[3] = { 1.0f, 1.0f, 1.0f };
        float g_light_intensity = 1.0f;

        float g_ambient_color[3] = { 1.0f, 1.0f, 1.0f };
        float g_ambient_intensity = 0.15f;

        struct ConeState {
            float g_position[3] = { 0.0f, 0.0f, 0.0f };
            float g_rotation_degrees[3] = { 0.0f, 0.0f, 0.0f };
            float g_scale[3] = { 1.0f, 1.0f, 1.0f };

            bool  g_anim_playing = true;
            float g_anim_speed = 1.0f;
            float g_anim_radius = 1.5f;
            float g_anim_height = 0.0f;
            float g_anim_rot_speed = 1.0f;
            float g_anim_time = 0.0f;

            float g_tint[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

            float g_specular_color[3] = { 0.25f, 0.25f, 0.25f };
            float g_shininess = 32.0f;

            float g_model[4][4] = {};
        };
        ConeState g_cones[CONE_COUNT];
        int g_selected_cone = 0;

        double g_last_time = 0.0;


        void mat4_identity(float m[4][4])
        {
            memset(m, 0, sizeof(float) * 16);
            m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0f;
        }

        void mat4_multiply(const float a[4][4], const float b[4][4], float out[4][4])
        {
            float r[4][4] = {};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < 4; ++k)
                        r[i][j] += a[k][j] * b[i][k];
            memcpy(out, r, sizeof(r));
        }

        void mat4_perspective(float fov_y, float aspect, float znear, float zfar, float m[4][4])
        {
            memset(m, 0, sizeof(float) * 16);
            const float tan_half = tanf(fov_y * 0.5f);
            m[0][0] = 1.0f / (aspect * tan_half);
            m[1][1] = 1.0f / tan_half;
            m[2][2] = zfar / (zfar - znear);
            m[2][3] = 1.0f;
            m[3][2] = -(zfar * znear) / (zfar - znear);
        }

        void mat4_ortho(float left, float right, float bottom, float top,
            float znear, float zfar, float m[4][4])
        {
            memset(m, 0, sizeof(float) * 16);
            m[0][0] = 2.0f / (right - left);
            m[1][1] = 2.0f / (bottom - top);
            m[2][2] = 1.0f / (zfar - znear);
            m[3][0] = -(right + left) / (right - left);
            m[3][1] = -(top + bottom) / (bottom - top);
            m[3][2] = -znear / (zfar - znear);
            m[3][3] = 1.0f;
        }

        void mat4_translation(float x, float y, float z, float m[4][4])
        {
            mat4_identity(m);
            m[3][0] = x;
            m[3][1] = y;
            m[3][2] = z;
        }

        void mat4_scale(float x, float y, float z, float m[4][4])
        {
            mat4_identity(m);
            m[0][0] = x;
            m[1][1] = y;
            m[2][2] = z;
        }

        void mat4_rotation_x(float angle, float m[4][4])
        {
            mat4_identity(m);
            const float c = cosf(angle), s = sinf(angle);
            m[1][1] = c;  m[2][1] = -s;
            m[1][2] = s;  m[2][2] = c;
        }

        void mat4_rotation_y(float angle, float m[4][4])
        {
            mat4_identity(m);
            const float c = cosf(angle), s = sinf(angle);
            m[0][0] = c;  m[2][0] = s;
            m[0][2] = -s; m[2][2] = c;
        }

        void mat4_rotation_z(float angle, float m[4][4])
        {
            mat4_identity(m);
            const float c = cosf(angle), s = sinf(angle);
            m[0][0] = c;  m[1][0] = -s;
            m[0][1] = s;  m[1][1] = c;
        }

        void updateCamera()
        {
            float Rx[4][4], Ry[4][4], Rz[4][4];
            float T[4][4], tmp[4][4], inverse_rotation[4][4];

            mat4_rotation_x(
                -g_camera.rotation_degrees[0] * PI / 180.0f, Rx);

            mat4_rotation_y(
                -g_camera.rotation_degrees[1] * PI / 180.0f, Ry);

            mat4_rotation_z(
                -g_camera.rotation_degrees[2] * PI / 180.0f, Rz);

            mat4_multiply(Rx, Ry, tmp);
            mat4_multiply(Rz, tmp, inverse_rotation);

            mat4_translation(
                -g_camera.position[0],
                -g_camera.position[1],
                -g_camera.position[2],
                T
            );

            mat4_multiply(inverse_rotation, T, g_view);

            const auto& extent = graphics::internal::context.swapchain_extent;

            if (extent.width == 0 || extent.height == 0) {
                return;
            }

            const float aspect =
                static_cast<float>(extent.width) /
                static_cast<float>(extent.height);

            if (g_camera.projection_mode == ProjectionMode::Perspective) {
                mat4_perspective(
                    g_camera.fov_degrees * PI / 180.0f,
                    aspect,
                    0.1f,
                    100.0f,
                    g_camera.projection
                );
            }
            else {
                const float half_h = g_camera.ortho_scale;
                const float half_w = half_h * aspect;

                mat4_ortho(
                    -half_w,
                    half_w,
                    half_h,
                    -half_h,
                    0.1f,
                    100.0f,
                    g_camera.projection
                );
            }
        }

        template<typename T>
        bool createMappedBuffer(
            VkDeviceSize size,
            VkBufferUsageFlags usage,
            VkBuffer& buffer,
            VmaAllocation& allocation,
            T*& memory,
            const void* initial_data = nullptr)
        {
            auto& ctx = graphics::internal::context;

            const VkBufferCreateInfo buffer_info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = size,
                .usage = usage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };

            const VmaAllocationCreateInfo allocation_info = {
                .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
            };

            VmaAllocationInfo memory_info{};

            if (vmaCreateBuffer(
                ctx.allocator,
                &buffer_info,
                &allocation_info,
                &buffer,
                &allocation,
                &memory_info) != VK_SUCCESS) {

                buffer = VK_NULL_HANDLE;
                allocation = nullptr;
                memory = nullptr;

                std::cerr << "Failed to create buffer\n";
                return false;
            }

            memory = static_cast<T*>(memory_info.pMappedData);

            if (!memory) {
                std::cerr << "Buffer memory is not mapped\n";

                vmaDestroyBuffer(ctx.allocator, buffer, allocation);
                buffer = VK_NULL_HANDLE;
                allocation = nullptr;
                return false;
            }

            if (initial_data) {
                std::memcpy(memory, initial_data, static_cast<size_t>(size));

                if (vmaFlushAllocation(
                    ctx.allocator, allocation, 0, size) != VK_SUCCESS) {

                    std::cerr << "Failed to flush buffer\n";

                    vmaDestroyBuffer(ctx.allocator, buffer, allocation);
                    buffer = VK_NULL_HANDLE;
                    allocation = nullptr;
                    memory = nullptr;
                    return false;
                }
            }

            return true;
        }

        bool createConeGeometry()
        {
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;

            const float normal_length = std::sqrt(
                CONE_HEIGHT * CONE_HEIGHT +
                CONE_RADIUS * CONE_RADIUS
            );

            for (uint32_t i = 0; i < CONE_SEGMENTS; ++i) {
                const float angle =
                    2.0f * PI * float(i) / float(CONE_SEGMENTS);

                const float c = std::cos(angle);
                const float s = std::sin(angle);

                Vertex v{};

                v.position[0] = CONE_RADIUS * c;
                v.position[2] = CONE_RADIUS * s;

                v.color[0] = c * 0.5f + 0.5f;
                v.color[2] = s * 0.5f + 0.5f;

                v.normal[0] = CONE_HEIGHT * c / normal_length;
                v.normal[1] = -CONE_RADIUS / normal_length;
                v.normal[2] = CONE_HEIGHT * s / normal_length;

                vertices.push_back(v);
            }

            const uint32_t apex_start =
                static_cast<uint32_t>(vertices.size());

            for (uint32_t i = 0; i < CONE_SEGMENTS; ++i) {
                vertices.push_back({
                    { 0.0f, -CONE_HEIGHT, 0.0f },
                    { 0.5f, 1.0f, 0.5f },
                    { 0.0f, -1.0f, 0.0f }
                    });
            }

            const uint32_t base_center =
                static_cast<uint32_t>(vertices.size());

            Vertex center{};

            center.color[0] = 0.5f;
            center.color[2] = 0.5f;
            center.normal[1] = 1.0f;

            vertices.push_back(center);

            const uint32_t base_start =
                static_cast<uint32_t>(vertices.size());

            for (uint32_t i = 0; i < CONE_SEGMENTS; ++i) {
                const float angle =
                    2.0f * PI * float(i) / float(CONE_SEGMENTS);

                const float c = std::cos(angle);
                const float s = std::sin(angle);

                Vertex v{};

                v.position[0] = CONE_RADIUS * c;
                v.position[2] = CONE_RADIUS * s;

                v.color[0] = c * 0.5f + 0.5f;
                v.color[2] = s * 0.5f + 0.5f;

                v.normal[1] = 1.0f;

                vertices.push_back(v);
            }

            for (uint32_t i = 0; i < CONE_SEGMENTS; ++i) {
                const uint32_t next = (i + 1) % CONE_SEGMENTS;

                indices.push_back(apex_start + i);
                indices.push_back(i);
                indices.push_back(next);

                indices.push_back(base_center);
                indices.push_back(base_start + next);
                indices.push_back(base_start + i);
            }

            vk_index_count = static_cast<uint32_t>(indices.size());

            if (!createMappedBuffer(
                vertices.size() * sizeof(Vertex),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                vk_vertex_buffer,
                vk_vertex_buffer_allocation,
                vk_vertex_buffer_memory,
                vertices.data())) {
                return false;
            }

            return createMappedBuffer(
                indices.size() * sizeof(uint32_t),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                vk_index_buffer,
                vk_index_buffer_allocation,
                vk_index_buffer_memory,
                indices.data()
            );
        }

        bool createUniformBuffer()
        {
            if (!createMappedBuffer(
                sizeof(SceneUniforms),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                vk_scene_uniform_buffer,
                vk_scene_uniform_allocation,
                vk_scene_uniform_memory)) {
                return false;
            }

            for (uint32_t i = 0; i < CONE_COUNT; ++i) {
                if (!createMappedBuffer(
                    sizeof(ModelUniforms),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    vk_uniform_buffer[i],
                    vk_uniform_buffer_allocation[i],
                    vk_uniform_buffer_memory[i])) {
                    return false;
                }
            }

            return true;
        }

        void destroyLightStorageBuffer(LightStorageBuffer& storage)
        {
            auto& ctx = graphics::internal::context;

            if (storage.buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(
                    ctx.allocator,
                    storage.buffer,
                    storage.allocation
                );
            }

            storage = LightStorageBuffer{};
        }

        template<typename T>
        bool createLightStorageBuffer(
            const std::vector<T>& lights,
            LightStorageBuffer& storage)
        {
            if (storage.buffer != VK_NULL_HANDLE) {
                std::cerr << "Light storage buffer already exists\n";
                return false;
            }

            const size_t count = lights.empty() ? 1 : lights.size();
            const VkDeviceSize size =
                sizeof(LightBufferHeader) + VkDeviceSize(count) * sizeof(T);

            if (!createMappedBuffer(
                size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                storage.buffer,
                storage.allocation,
                storage.memory)) {
                return false;
            }

            storage.size = size;
            std::memset(storage.memory, 0, static_cast<size_t>(size));

            const LightBufferHeader header = {
                static_cast<uint32_t>(lights.size()),
                { 0, 0, 0 }
            };

            std::memcpy(storage.memory, &header, sizeof(header));

            if (!lights.empty()) {
                auto* data = static_cast<unsigned char*>(storage.memory)
                    + sizeof(header);

                std::memcpy(data, lights.data(), lights.size() * sizeof(T));
            }

            if (vmaFlushAllocation(
                graphics::internal::context.allocator,
                storage.allocation, 0, size) != VK_SUCCESS) {
                std::cerr << "Failed to flush light storage buffer\n";
                destroyLightStorageBuffer(storage);
                return false;
            }

            return true;
        }

        bool createLightStorageBuffers()
        {
            return createLightStorageBuffer(g_point_lights, g_point_light_buffer)
                && createLightStorageBuffer(g_spot_lights, g_spot_light_buffer);
        }

        bool updateLightStorageBuffers()
        {
            if (!g_lights_dirty) {
                return true;
            }

            auto& ctx = graphics::internal::context;

            LightStorageBuffer new_points;
            LightStorageBuffer new_spots;

            if (!createLightStorageBuffer(g_point_lights, new_points)) {
                return false;
            }

            if (!createLightStorageBuffer(g_spot_lights, new_spots)) {
                destroyLightStorageBuffer(new_points);
                return false;
            }

            if (vkQueueWaitIdle(ctx.graphics_queue) != VK_SUCCESS) {
                destroyLightStorageBuffer(new_points);
                destroyLightStorageBuffer(new_spots);
                return false;
            }

            const VkDescriptorBufferInfo buffer_infos[] = {
                {
                    .buffer = new_points.buffer,
                    .offset = 0,
                    .range = new_points.size,
                },
                {
                    .buffer = new_spots.buffer,
                    .offset = 0,
                    .range = new_spots.size,
                },
            };

            const VkWriteDescriptorSet writes[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = vk_scene_descriptor_set,
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffer_infos[0],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = vk_scene_descriptor_set,
                    .dstBinding = 2,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffer_infos[1],
                },
            };

            vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);

            destroyLightStorageBuffer(g_point_light_buffer);
            destroyLightStorageBuffer(g_spot_light_buffer);

            g_point_light_buffer = new_points;
            g_spot_light_buffer = new_spots;

            g_lights_dirty = false;
            return true;
        }

        void drawLightControls()
        {
            if (ImGui::CollapsingHeader(
                "Point lights",
                ImGuiTreeNodeFlags_DefaultOpen)) {

                if (ImGui::Button("Add point light")) {
                    g_point_lights.push_back({
                        .position = { 1.5f, -0.5f, -2.0f, 1.0f },
                        .color_intensity = { 1.0f, 1.0f, 1.0f, 4.0f }
                    });
                    g_lights_dirty = true;
                }

                ImGui::Text(
                    "Count: %d",
                    static_cast<int>(g_point_lights.size())
                );

                ImGui::PushID("point_lights");

                for (size_t i = 0; i < g_point_lights.size();) {
                    ImGui::PushID(static_cast<int>(i));

                    auto& light = g_point_lights[i];

                    ImGui::Separator();
                    ImGui::Text("Point light %d", static_cast<int>(i + 1));

                    if (ImGui::Button("Delete")) {
                        g_point_lights.erase(
                            g_point_lights.begin() +
                            static_cast<std::ptrdiff_t>(i)
                        );

                        g_lights_dirty = true;
                        ImGui::PopID();
                        continue;
                    }

                    if (ImGui::DragFloat3("Position", light.position, 0.02f)) {
                        g_lights_dirty = true;
                    }

                    if (ImGui::ColorEdit3("Color", light.color_intensity)) {
                        g_lights_dirty = true;
                    }

                    if (ImGui::SliderFloat("Intensity", &light.color_intensity[3], 0.0f, 30.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
                        g_lights_dirty = true;
                    }

                    ImGui::PopID();
                    ++i;
                }

                ImGui::PopID();
            }

            if (ImGui::CollapsingHeader(
                "Spot lights",
                ImGuiTreeNodeFlags_DefaultOpen)) {

                if (ImGui::Button("Add spot light")) {
                    g_spot_lights.push_back({
                        .position = { 1.5f, -0.5f, -2.0f, 1.0f },
                        .direction = { 0.0f, 0.0f, 1.0f, 0.0f },
                        .color_intensity = { 1.0f, 1.0f, 1.0f, 4.0f },
                        .cone_angles = {
                            std::cos(15.0f * PI / 180.0f),
                            std::cos(25.0f * PI / 180.0f),
                            0.0f, 0.0f
                        }
                        });

                    g_lights_dirty = true;
                }

                ImGui::Text(
                    "Count: %d",
                    static_cast<int>(g_spot_lights.size())
                );

                ImGui::PushID("spot_lights");

                for (size_t i = 0; i < g_spot_lights.size();) {
                    ImGui::PushID(static_cast<int>(i));

                    auto& light = g_spot_lights[i];

                    ImGui::Separator();
                    ImGui::Text("Spot light %d", static_cast<int>(i + 1));

                    if (ImGui::Button("Delete")) {
                        g_spot_lights.erase(
                            g_spot_lights.begin() +
                            static_cast<std::ptrdiff_t>(i)
                        );

                        g_lights_dirty = true;
                        ImGui::PopID();
                        continue;
                    }

                    if (ImGui::DragFloat3("Position", light.position, 0.02f)) {
                        g_lights_dirty = true;
                    }

                    if (ImGui::DragFloat3("Ray direction", light.direction, 0.02f)) {
                        g_lights_dirty = true;
                    }

                    const float direction_length_squared =
                        light.direction[0] * light.direction[0] +
                        light.direction[1] * light.direction[1] +
                        light.direction[2] * light.direction[2];

                    if (direction_length_squared < 0.00000001f) {
                        ImGui::TextUnformatted(
                            "Set a non-zero ray direction."
                        );
                    }

                    if (ImGui::ColorEdit3("Color", light.color_intensity)) {
                        g_lights_dirty = true;
                    }

                    if (ImGui::SliderFloat("Intensity", &light.color_intensity[3], 0.0f, 30.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
                        g_lights_dirty = true;
                    }

                    const float inner_cos = std::fmax(
                        -1.0f, std::fmin(1.0f, light.cone_angles[0]));

                    const float outer_cos = std::fmax(
                        -1.0f, std::fmin(1.0f, light.cone_angles[1]));

                    float inner_degrees =
                        std::acos(inner_cos) * 180.0f / PI;

                    float outer_degrees =
                        std::acos(outer_cos) * 180.0f / PI;

                    bool angles_changed = false;

                    if (ImGui::SliderFloat("Inner half-angle (deg)", &inner_degrees, 0.0f, 89.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp)) {

                        if (inner_degrees > outer_degrees) {
                            outer_degrees = inner_degrees;
                        }

                        angles_changed = true;
                    }

                    if (ImGui::SliderFloat("Outer half-angle (deg)", &outer_degrees, 0.0f, 89.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp)) {

                        if (outer_degrees < inner_degrees) {
                            inner_degrees = outer_degrees;
                        }

                        angles_changed = true;
                    }

                    if (angles_changed) {
                        light.cone_angles[0] =
                            std::cos(inner_degrees * PI / 180.0f);

                        light.cone_angles[1] =
                            std::cos(outer_degrees * PI / 180.0f);

                        g_lights_dirty = true;
                    }

                    ImGui::PopID();
                    ++i;
                }

                ImGui::PopID();
            }
        }

        bool createDescriptorSet()
        {
            auto& ctx = graphics::internal::context;

            const VkDescriptorSetLayoutBinding scene_bindings[] = {
                {
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 2,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
            };

            const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 3,
                .pBindings = scene_bindings,
            };

            if (vkCreateDescriptorSetLayout(
                ctx.device,
                &scene_layout_info,
                nullptr,
                &vk_scene_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create scene descriptor layout\n";
                return false;
            }

            const VkDescriptorSetLayoutBinding model_binding = {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                              VK_SHADER_STAGE_FRAGMENT_BIT,
            };

            const VkDescriptorSetLayoutCreateInfo model_layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 1,
                .pBindings = &model_binding,
            };

            if (vkCreateDescriptorSetLayout(
                ctx.device,
                &model_layout_info,
                nullptr,
                &vk_model_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create model descriptor layout\n";
                return false;
            }

            const VkDescriptorPoolSize pool_sizes[] = {
                {
                    .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = CONE_COUNT + 1,
                },
                {
                    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 2,
                },
            };

            const VkDescriptorPoolCreateInfo pool_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = CONE_COUNT + 1,
                .poolSizeCount = 2,
                .pPoolSizes = pool_sizes,
            };

            if (vkCreateDescriptorPool(
                ctx.device,
                &pool_info,
                nullptr,
                &vk_descriptor_pool) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor pool\n";
                return false;
            }

            const auto allocate_set = [&](
                VkDescriptorSetLayout layout,
                VkDescriptorSet& descriptor_set) -> bool
                {
                    const VkDescriptorSetAllocateInfo alloc_info = {
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        .descriptorPool = vk_descriptor_pool,
                        .descriptorSetCount = 1,
                        .pSetLayouts = &layout,
                    };

                    if (vkAllocateDescriptorSets(
                        ctx.device,
                        &alloc_info,
                        &descriptor_set) != VK_SUCCESS) {
                        std::cerr << "Failed to allocate descriptor set\n";
                        return false;
                    }

                    return true;
                };

            const auto write_buffer = [&](
                VkDescriptorSet descriptor_set,
                uint32_t binding,
                VkDescriptorType type,
                VkBuffer buffer,
                VkDeviceSize size)
                {
                    const VkDescriptorBufferInfo buffer_info = {
                        .buffer = buffer,
                        .offset = 0,
                        .range = size,
                    };

                    const VkWriteDescriptorSet write = {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .dstSet = descriptor_set,
                        .dstBinding = binding,
                        .descriptorCount = 1,
                        .descriptorType = type,
                        .pBufferInfo = &buffer_info,
                    };

                    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
                };

            if (!allocate_set(
                vk_scene_set_layout,
                vk_scene_descriptor_set)) {
                return false;
            }

            write_buffer(
                vk_scene_descriptor_set,
                0,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                vk_scene_uniform_buffer,
                sizeof(SceneUniforms)
            );

            write_buffer(
                vk_scene_descriptor_set,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                g_point_light_buffer.buffer,
                g_point_light_buffer.size
            );

            write_buffer(
                vk_scene_descriptor_set,
                2,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                g_spot_light_buffer.buffer,
                g_spot_light_buffer.size
            );

            for (uint32_t i = 0; i < CONE_COUNT; ++i) {
                if (!allocate_set(
                    vk_model_set_layout,
                    vk_descriptor_set[i])) {
                    return false;
                }

                write_buffer(
                    vk_descriptor_set[i],
                    0,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    vk_uniform_buffer[i],
                    sizeof(ModelUniforms)
                );
            }

            return true;
        }

        VkShaderModule loadShaderModule(const char* path)
        {
            FILE* file = fopen(path, "rb");
            if (!file) {
                std::cerr << "Failed to open shader file: " << path << '\n';
                return VK_NULL_HANDLE;
            }
            fseek(file, 0, SEEK_END);
            const long size = ftell(file);
            fseek(file, 0, SEEK_SET);
            std::vector<char> buffer(static_cast<size_t>(size));
            fread(buffer.data(), 1, static_cast<size_t>(size), file);
            fclose(file);

            const VkShaderModuleCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = static_cast<size_t>(size),
                .pCode = reinterpret_cast<const uint32_t*>(buffer.data()),
            };
            VkShaderModule module = VK_NULL_HANDLE;
            if (vkCreateShaderModule(graphics::internal::context.device, &info,
                nullptr, &module) != VK_SUCCESS) {
                std::cerr << "Failed to create shader module: " << path << '\n';
                return VK_NULL_HANDLE;
            }
            return module;
        }

        bool createPipeline()
        {
            auto& ctx = graphics::internal::context;

            vk_vertex_shader = loadShaderModule(VERT_SHADER_PATH);
            vk_fragment_shader = loadShaderModule(FRAG_SHADER_PATH);
            if (!vk_vertex_shader || !vk_fragment_shader) return false;

            const VkPipelineShaderStageCreateInfo stages[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vk_vertex_shader,
                    .pName = "main",
                },
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = vk_fragment_shader,
                    .pName = "main",
                },
            };

            const VkVertexInputBindingDescription vertex_binding = {
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };
            const VkVertexInputAttributeDescription vertex_attributes[] = {
                {
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, position),
                },
                {
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, color),
                },
                {
                    .location = 2,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, normal),
                },
            };
            const VkPipelineVertexInputStateCreateInfo vertex_input = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &vertex_binding,
                .vertexAttributeDescriptionCount = 3,
                .pVertexAttributeDescriptions = vertex_attributes,
            };

            const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            const VkPipelineViewportStateCreateInfo viewport_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1,
                .scissorCount = 1,
            };

            const VkPipelineRasterizationStateCreateInfo raster = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .lineWidth = 1.0f,
            };

            const VkPipelineMultisampleStateCreateInfo multisample = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            };

            const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            const VkPipelineColorBlendAttachmentState color_attachment = {
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            };
            const VkPipelineColorBlendStateCreateInfo color_blend = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .attachmentCount = 1,
                .pAttachments = &color_attachment,
            };

            const VkDynamicState dynamic_states[] = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR,
            };
            const VkPipelineDynamicStateCreateInfo dynamic_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = 2,
                .pDynamicStates = dynamic_states,
            };

            const VkDescriptorSetLayout set_layouts[] = {
                vk_scene_set_layout,
                vk_model_set_layout,
            };

            const VkPipelineLayoutCreateInfo layout_info = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2,
                .pSetLayouts = set_layouts,
            };
            if (vkCreatePipelineLayout(ctx.device, &layout_info, nullptr,
                &vk_pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create pipeline layout\n";
                return false;
            }

            const VkGraphicsPipelineCreateInfo pipeline_info = {
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2,
                .pStages = stages,
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &input_assembly,
                .pViewportState = &viewport_state,
                .pRasterizationState = &raster,
                .pMultisampleState = &multisample,
                .pDepthStencilState = &depth_stencil,
                .pColorBlendState = &color_blend,
                .pDynamicState = &dynamic_state,
                .layout = vk_pipeline_layout,
                .renderPass = ctx.render_pass,
                .subpass = 0,
            };

            if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info,
                nullptr, &vk_pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create graphics pipeline\n";
                return false;
            }

            return true;
        }

    }

    bool initialize()
    {
        if (!createConeGeometry() ||
            !createUniformBuffer() ||
            !createLightStorageBuffers() ||
            !createDescriptorSet() ||
            !createPipeline()) {
            shutdown();
            return false;
        }

        for (auto& cone : g_cones) mat4_identity(cone.g_model);
        g_cones[1].g_position[0] = -1.25f;
        g_cones[1].g_tint[0] = 1.0f;
        g_cones[1].g_tint[1] = 0.4f;
        g_cones[1].g_tint[2] = 0.4f;
        g_cones[1].g_anim_playing = false;
        g_cones[2].g_position[2] = 0.75f;
        g_cones[2].g_tint[0] = 0.4f;
        g_cones[2].g_tint[1] = 0.6f;
        g_cones[2].g_tint[2] = 1.0f;
        g_cones[2].g_anim_playing = false;
        updateCamera();

        g_last_time = 0.0;

        return true;
    }

    void shutdown()
    {
        auto& ctx = graphics::internal::context;
        vkQueueWaitIdle(ctx.graphics_queue);

        vkDestroyPipeline(ctx.device, vk_pipeline, nullptr);
        vkDestroyPipelineLayout(ctx.device, vk_pipeline_layout, nullptr);
        vkDestroyShaderModule(ctx.device, vk_vertex_shader, nullptr);
        vkDestroyShaderModule(ctx.device, vk_fragment_shader, nullptr);

        vkDestroyDescriptorPool(ctx.device, vk_descriptor_pool, nullptr);
        vk_descriptor_pool = VK_NULL_HANDLE;
        vk_scene_descriptor_set = VK_NULL_HANDLE;

        destroyLightStorageBuffer(g_point_light_buffer);
        destroyLightStorageBuffer(g_spot_light_buffer);

        vkDestroyDescriptorSetLayout(
            ctx.device, vk_scene_set_layout, nullptr);
        vk_scene_set_layout = VK_NULL_HANDLE;

        vkDestroyDescriptorSetLayout(
            ctx.device, vk_model_set_layout, nullptr);
        vk_model_set_layout = VK_NULL_HANDLE;

        if (vk_scene_uniform_buffer) {
            vmaDestroyBuffer(
                ctx.allocator,
                vk_scene_uniform_buffer,
                vk_scene_uniform_allocation
            );

            vk_scene_uniform_buffer = VK_NULL_HANDLE;
            vk_scene_uniform_allocation = nullptr;
            vk_scene_uniform_memory = nullptr;
        }

        for (uint32_t i = 0; i < CONE_COUNT; ++i) {
            if (vk_uniform_buffer[i]) {
                vmaDestroyBuffer(ctx.allocator, vk_uniform_buffer[i], vk_uniform_buffer_allocation[i]);
                vk_uniform_buffer[i] = VK_NULL_HANDLE;
                vk_uniform_buffer_allocation[i] = nullptr;
                vk_uniform_buffer_memory[i] = nullptr;
            }
        }
        if (vk_index_buffer) vmaDestroyBuffer(ctx.allocator, vk_index_buffer, vk_index_buffer_allocation);
        if (vk_vertex_buffer) vmaDestroyBuffer(ctx.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
    }

    void update(double time)
    {
        const double dt = (g_last_time == 0.0) ? 0.0 : (time - g_last_time);
        g_last_time = time;

        ImGui::Begin("Lab 2 - Camera and Lighting");

        ImGui::SeparatorText("Directional light");

        ImGui::DragFloat3("Light ray direction", g_light_direction, 0.02f
        );

        ImGui::ColorEdit3("Light color", g_light_color);

        ImGui::SliderFloat("Light intensity", &g_light_intensity, 0.0f, 5.0f);

        ImGui::SeparatorText("Ambient light");

        ImGui::ColorEdit3("Ambient color", g_ambient_color);

        ImGui::SliderFloat("Ambient intensity", &g_ambient_intensity, 0.0f, 1.0f);

        drawLightControls();

        ImGui::TextUnformatted("Select cone:");
        ImGui::RadioButton("Cone 1", &g_selected_cone, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Cone 2", &g_selected_cone, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Cone 3", &g_selected_cone, 2);
        auto& selected = g_cones[g_selected_cone];

        ImGui::SeparatorText("Camera");

        ImGui::SliderFloat3("Camera position", g_camera.position, -10.0f, 10.0f);

        ImGui::SliderFloat3("Camera rotation (deg)", g_camera.rotation_degrees, -180.0f, 180.0f);

        int camera_proj_mode =
            static_cast<int>(g_camera.projection_mode);

        ImGui::RadioButton("Perspective", &camera_proj_mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &camera_proj_mode, 1);

        g_camera.projection_mode =
            static_cast<ProjectionMode>(camera_proj_mode);

        if (g_camera.projection_mode == ProjectionMode::Perspective) {
            ImGui::SliderFloat("Camera FOV (deg)", &g_camera.fov_degrees, 20.0f, 120.0f);
        }
        else {
            ImGui::SliderFloat("Camera ortho scale", &g_camera.ortho_scale, 0.2f, 5.0f);
        }

        if (ImGui::Button("Reset camera")) {
            g_camera = CameraState{};
        }

        ImGui::SeparatorText("Transform");
        ImGui::SliderFloat3("Position", selected.g_position, -3.0f, 3.0f);
        ImGui::SliderFloat3("Rotation (deg)", selected.g_rotation_degrees, -180.0f, 180.0f);
        ImGui::SliderFloat3("Scale", selected.g_scale, 0.1f, 3.0f);

        ImGui::SeparatorText("Animation");
        ImGui::Checkbox("Playing", &selected.g_anim_playing);
        ImGui::SliderFloat("Speed", &selected.g_anim_speed, 0.0f, 5.0f);
        ImGui::SliderFloat("Trajectory radius", &selected.g_anim_radius, 0.0f, 5.0f);
        ImGui::SliderFloat("Trajectory height", &selected.g_anim_height, -2.0f, 2.0f);
        ImGui::SliderFloat("Self-rotation speed", &selected.g_anim_rot_speed, 0.0f, 5.0f);

        if (ImGui::Button("Reset animation time")) {
            selected.g_anim_time = 0.0f;
        }

        ImGui::SeparatorText("Material");

        ImGui::ColorEdit3("Albedo multiplier", selected.g_tint);

        ImGui::ColorEdit3("Specular color", selected.g_specular_color);

        ImGui::SliderFloat("Shininess", &selected.g_shininess, 1.0f, 256.0f);

        ImGui::End();

        for (auto& cone : g_cones) {
            if (cone.g_anim_playing) {
                cone.g_anim_time += float(dt) * cone.g_anim_speed;
            }

            const float t = cone.g_anim_time;

            float T[4][4], R[4][4], S[4][4];
            float Rx[4][4], Ry[4][4], Rz[4][4], tmp1[4][4];

            const float px = cone.g_position[0] + cone.g_anim_radius * cosf(t);

            const float py = cone.g_position[1] + cone.g_anim_height;

            const float pz = cone.g_position[2] + cone.g_anim_radius * sinf(t);

            mat4_translation(px, py, pz, T);

            const float anim_rot_y = t * cone.g_anim_rot_speed;

            mat4_rotation_x( cone.g_rotation_degrees[0] * PI / 180.0f, Rx);

            mat4_rotation_y( cone.g_rotation_degrees[1] * PI / 180.0f + anim_rot_y, Ry);

            mat4_rotation_z( cone.g_rotation_degrees[2] * PI / 180.0f, Rz);

            mat4_multiply(Ry, Rx, tmp1);
            mat4_multiply(tmp1, Rz, R);

            mat4_scale(
                cone.g_scale[0],
                cone.g_scale[1],
                cone.g_scale[2],
                S
            );

            mat4_multiply(R, S, tmp1);
            mat4_multiply(T, tmp1, cone.g_model);
        }

        updateCamera();

    }

    void render(const graphics::internal::FrameData& fd)
    {
        auto& ctx = graphics::internal::context;

        if (!updateLightStorageBuffers()) {
            std::cerr << "Failed to update light buffers\n";
        }

        auto& scene = *vk_scene_uniform_memory;

        mat4_multiply(
            g_camera.projection,
            g_view,
            scene.view_projection
        );

        for (int component = 0; component < 3; ++component) {
            scene.camera_position[component] =
                g_camera.position[component];

            scene.light_direction[component] =
                g_light_direction[component];

            scene.light_color_intensity[component] =
                g_light_color[component];

            scene.ambient_color_intensity[component] =
                g_ambient_color[component];
        }

        scene.camera_position[3] = 1.0f;
        scene.light_direction[3] = 0.0f;
        scene.light_color_intensity[3] = g_light_intensity;
        scene.ambient_color_intensity[3] = g_ambient_intensity;

        if (vmaFlushAllocation(
            ctx.allocator,
            vk_scene_uniform_allocation,
            0,
            sizeof(SceneUniforms)) != VK_SUCCESS) {
            std::cerr << "Failed to flush scene uniform buffer\n";
        }

        for (uint32_t i = 0; i < CONE_COUNT; ++i) {
            auto& model = *vk_uniform_buffer_memory[i];

            memcpy(
                model.model,
                g_cones[i].g_model,
                sizeof(model.model)
            );

            memcpy(
                model.tint,
                g_cones[i].g_tint,
                sizeof(model.tint)
            );

            for (int component = 0; component < 3; ++component) {
                model.specular_shininess[component] =
                    g_cones[i].g_specular_color[component];
            }

            model.specular_shininess[3] = g_cones[i].g_shininess;

            if (vmaFlushAllocation(
                ctx.allocator,
                vk_uniform_buffer_allocation[i],
                0,
                sizeof(ModelUniforms)) != VK_SUCCESS) {
                std::cerr << "Failed to flush model uniform buffer\n";
            }
        }

        vkResetCommandBuffer(fd.command_buffer, 0);

        const VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(fd.command_buffer, &begin_info);

        const VkClearValue clear_values[] = {
            {.color = {.float32 = { 0.1f, 0.1f, 0.1f, 1.0f } } },
            {.depthStencil = {.depth = 1.0f, .stencil = 0 } },
        };
        const VkRenderPassBeginInfo render_pass_begin = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = ctx.render_pass,
            .framebuffer = fd.framebuffer,
            .renderArea = {.offset = { 0, 0 }, .extent = ctx.swapchain_extent },
            .clearValueCount = 2,
            .pClearValues = clear_values,
        };
        vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin,
            VK_SUBPASS_CONTENTS_INLINE);

        const VkViewport viewport = {
            .x = 0.0f,
            .y = 0.0f,
            .width = float(ctx.swapchain_extent.width),
            .height = float(ctx.swapchain_extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        const VkRect2D scissor = { .extent = ctx.swapchain_extent };
        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

        const VkDeviceSize vertex_offset = 0;
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &vertex_offset);
        vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(
            fd.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_pipeline_layout,
            0,
            1,
            &vk_scene_descriptor_set,
            0,
            nullptr
        );

        for (uint32_t i = 0; i < CONE_COUNT; ++i) {
            vkCmdBindDescriptorSets(
                fd.command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_pipeline_layout,
                1,
                1,
                &vk_descriptor_set[i],
                0,
                nullptr
            );

            vkCmdDrawIndexed(
                fd.command_buffer,
                vk_index_count,
                1,
                0,
                0,
                0
            );
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application