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
        };

        struct GlobalUniforms {
            float model[4][4];
            float view[4][4];
            float projection[4][4];
            float tint[4];
        };

        // ===== Буферы =====
        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;
        Vertex* vk_vertex_buffer_memory = nullptr;

        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;
        uint32_t* vk_index_buffer_memory = nullptr;
        uint32_t vk_index_count = 0;

        VkBuffer vk_uniform_buffer[CONE_COUNT] = {};
        VmaAllocation vk_uniform_buffer_allocation[CONE_COUNT] = {};
        GlobalUniforms* vk_uniform_buffer_memory[CONE_COUNT] = {};

        VkDescriptorSetLayout vk_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet vk_descriptor_set[CONE_COUNT] = {};

        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline vk_pipeline = VK_NULL_HANDLE;

        VkShaderModule vk_vertex_shader = VK_NULL_HANDLE;
        VkShaderModule vk_fragment_shader = VK_NULL_HANDLE;

        float g_view[4][4];

        enum class ProjectionMode { Perspective, Orthographic };

        // Each object has its own parameters; formulas are unchanged.
        struct ConeState {
            ProjectionMode projection_mode = ProjectionMode::Perspective;
            float fov_degrees = 60.0f;
            float ortho_scale = 1.5f;
            float projection[4][4] = {};
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

        void mat4_look_at_origin(float distance, float m[4][4])
        {
            mat4_identity(m);
            m[3][2] = distance;
        }


        bool createConeGeometry()
        {
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;

            {
                Vertex v{};
                v.position[0] = 0.0f;
                v.position[1] = -CONE_HEIGHT;
                v.position[2] = 0.0f;
                v.color[0] = 0.5f;
                v.color[1] = 1.0f;
                v.color[2] = 0.5f;
                vertices.push_back(v);
            }

            for (uint32_t i = 0; i < CONE_SEGMENTS; ++i) {
                const float angle = 2.0f * PI * float(i) / float(CONE_SEGMENTS);
                const float x = CONE_RADIUS * cosf(angle);
                const float y = 0.0f;
                const float z = CONE_RADIUS * sinf(angle);

                Vertex v{};
                v.position[0] = x;
                v.position[1] = y;
                v.position[2] = z;
                v.color[0] = (x / CONE_RADIUS) * 0.5f + 0.5f;
                v.color[1] = 0.0f;
                v.color[2] = (z / CONE_RADIUS) * 0.5f + 0.5f;
                vertices.push_back(v);
            }

            {
                Vertex v{};
                v.position[0] = 0.0f;
                v.position[1] = 0.0f;
                v.position[2] = 0.0f;
                v.color[0] = 0.5f;
                v.color[1] = 0.0f;
                v.color[2] = 0.5f;
                vertices.push_back(v);
            }

            const uint32_t apex_index = 0;
            const uint32_t base_start = 1;
            const uint32_t base_center = 1 + CONE_SEGMENTS;

            for (uint32_t i = 0; i < CONE_SEGMENTS; ++i) {
                const uint32_t i0 = base_start + i;
                const uint32_t i1 = base_start + ((i + 1) % CONE_SEGMENTS);
                indices.push_back(apex_index);
                indices.push_back(i1);
                indices.push_back(i0);
            }

            for (uint32_t i = 0; i < CONE_SEGMENTS; ++i) {
                const uint32_t i0 = base_start + i;
                const uint32_t i1 = base_start + ((i + 1) % CONE_SEGMENTS);
                indices.push_back(base_center);
                indices.push_back(i0);
                indices.push_back(i1);
            }

            vk_index_count = static_cast<uint32_t>(indices.size());

            auto& ctx = graphics::internal::context;

            {
                const VkBufferCreateInfo info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(Vertex) * vertices.size(),
                    .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                const VmaAllocationCreateInfo alloc = {
                    .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO,
                };
                VmaAllocationInfo alloc_info{};
                if (vmaCreateBuffer(ctx.allocator, &info, &alloc,
                    &vk_vertex_buffer, &vk_vertex_buffer_allocation,
                    &alloc_info) != VK_SUCCESS) {
                    std::cerr << "Failed to create vertex buffer\n";
                    return false;
                }
                vk_vertex_buffer_memory = static_cast<Vertex*>(alloc_info.pMappedData);
                if (!vk_vertex_buffer_memory) return false;
                memcpy(vk_vertex_buffer_memory, vertices.data(),
                    sizeof(Vertex) * vertices.size());
                if (vmaFlushAllocation(ctx.allocator, vk_vertex_buffer_allocation,
                    0, VK_WHOLE_SIZE) != VK_SUCCESS) return false;
            }

            {
                const VkBufferCreateInfo info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(uint32_t) * indices.size(),
                    .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                const VmaAllocationCreateInfo alloc = {
                    .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO,
                };
                VmaAllocationInfo alloc_info{};
                if (vmaCreateBuffer(ctx.allocator, &info, &alloc,
                    &vk_index_buffer, &vk_index_buffer_allocation,
                    &alloc_info) != VK_SUCCESS) {
                    std::cerr << "Failed to create index buffer\n";
                    return false;
                }
                vk_index_buffer_memory = static_cast<uint32_t*>(alloc_info.pMappedData);
                if (!vk_index_buffer_memory) return false;
                memcpy(vk_index_buffer_memory, indices.data(),
                    sizeof(uint32_t) * indices.size());
                if (vmaFlushAllocation(ctx.allocator, vk_index_buffer_allocation,
                    0, VK_WHOLE_SIZE) != VK_SUCCESS) return false;
            }

            return true;
        }

        bool createUniformBuffer()
        {
            auto& ctx = graphics::internal::context;

            const VkBufferCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = sizeof(GlobalUniforms),
                .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            const VmaAllocationCreateInfo alloc = {
                .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
            };
            for (uint32_t i = 0; i < CONE_COUNT; ++i) {
                VmaAllocationInfo alloc_info{};
                if (vmaCreateBuffer(ctx.allocator, &info, &alloc,
                    &vk_uniform_buffer[i], &vk_uniform_buffer_allocation[i],
                    &alloc_info) != VK_SUCCESS) {
                    std::cerr << "Failed to create uniform buffer\n";
                    return false;
                }
                vk_uniform_buffer_memory[i] = static_cast<GlobalUniforms*>(alloc_info.pMappedData);
                if (!vk_uniform_buffer_memory[i]) return false;
            }
            return true;
        }


        bool createDescriptorSet()
        {
            auto& ctx = graphics::internal::context;

            const VkDescriptorSetLayoutBinding binding = {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            };

            const VkDescriptorSetLayoutCreateInfo layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 1,
                .pBindings = &binding,
            };
            if (vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr,
                &vk_descriptor_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor set layout\n";
                return false;
            }

            const VkDescriptorPoolSize pool_size = {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = CONE_COUNT,
            };
            const VkDescriptorPoolCreateInfo pool_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = CONE_COUNT,
                .poolSizeCount = 1,
                .pPoolSizes = &pool_size,
            };
            if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr,
                &vk_descriptor_pool) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor pool\n";
                return false;
            }

            for (uint32_t i = 0; i < CONE_COUNT; ++i) {
                const VkDescriptorSetAllocateInfo alloc_info = {
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .descriptorPool = vk_descriptor_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &vk_descriptor_set_layout,
                };
                if (vkAllocateDescriptorSets(ctx.device, &alloc_info,
                    &vk_descriptor_set[i]) != VK_SUCCESS) {
                    std::cerr << "Failed to allocate descriptor set\n";
                    return false;
                }

                const VkDescriptorBufferInfo buffer_info = {
                    .buffer = vk_uniform_buffer[i],
                    .offset = 0,
                    .range = sizeof(GlobalUniforms),
                };
                const VkWriteDescriptorSet write = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = vk_descriptor_set[i],
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_info,
                };
                vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);

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
            };
            const VkPipelineVertexInputStateCreateInfo vertex_input = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &vertex_binding,
                .vertexAttributeDescriptionCount = 2,
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

            const VkPipelineLayoutCreateInfo layout_info = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &vk_descriptor_set_layout,
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
        if (!createConeGeometry() || !createUniformBuffer() ||
            !createDescriptorSet() || !createPipeline()) {
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
        mat4_look_at_origin(3.0f, g_view);

        const float aspect = float(graphics::internal::context.swapchain_extent.width) /
            float(graphics::internal::context.swapchain_extent.height);
        for (auto& cone : g_cones) {
            mat4_perspective(
                cone.fov_degrees * PI / 180.0f,
                aspect,
                0.1f,
                100.0f,
                cone.projection
            );
        }

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
        vkDestroyDescriptorSetLayout(ctx.device, vk_descriptor_set_layout, nullptr);

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

        ImGui::Begin("Lab 1 - Cone (Variant 9)");

        ImGui::TextUnformatted("Select cone:");
        ImGui::RadioButton("Cone 1", &g_selected_cone, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Cone 2", &g_selected_cone, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Cone 3", &g_selected_cone, 2);
        auto& selected = g_cones[g_selected_cone];

        ImGui::SeparatorText("Projection");

        int proj_mode = static_cast<int>(selected.projection_mode);

        ImGui::RadioButton("Perspective", &proj_mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &proj_mode, 1);

        selected.projection_mode = static_cast<ProjectionMode>(proj_mode);

        if (selected.projection_mode == ProjectionMode::Perspective) {
            ImGui::SliderFloat(
                "FOV (deg)",
                &selected.fov_degrees,
                20.0f,
                120.0f
            );
        }
        else {
            ImGui::SliderFloat(
                "Ortho scale",
                &selected.ortho_scale,
                0.2f,
                5.0f
            );
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

        ImGui::SeparatorText("Color");
        ImGui::ColorEdit3("Tint", selected.g_tint);

        ImGui::End();

        for (auto& cone : g_cones) {
            // Только изменение времени зависит от Playing.
            if (cone.g_anim_playing) {
                cone.g_anim_time += float(dt) * cone.g_anim_speed;
            }

            const float t = cone.g_anim_time;

            float T[4][4], R[4][4], S[4][4];
            float Rx[4][4], Ry[4][4], Rz[4][4], tmp1[4][4];

            // Положение вычисляется и при паузе — по замороженному времени.
            const float px =
                cone.g_position[0] + cone.g_anim_radius * cosf(t);

            const float py =
                cone.g_position[1] + cone.g_anim_height;

            const float pz =
                cone.g_position[2] + cone.g_anim_radius * sinf(t);

            mat4_translation(px, py, pz, T);

            const float anim_rot_y = t * cone.g_anim_rot_speed;

            mat4_rotation_x(
                cone.g_rotation_degrees[0] * PI / 180.0f, Rx);

            mat4_rotation_y(
                cone.g_rotation_degrees[1] * PI / 180.0f + anim_rot_y, Ry);

            mat4_rotation_z(
                cone.g_rotation_degrees[2] * PI / 180.0f, Rz);

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

        const float aspect = float(graphics::internal::context.swapchain_extent.width) /
            float(graphics::internal::context.swapchain_extent.height);

        for (auto& cone : g_cones) {
            if (cone.projection_mode == ProjectionMode::Perspective) {
                mat4_perspective(
                    cone.fov_degrees * PI / 180.0f,
                    aspect,
                    0.1f,
                    100.0f,
                    cone.projection
                );
            }
            else {
                const float half_h = cone.ortho_scale;
                const float half_w = half_h * aspect;

                mat4_ortho(
                    -half_w,
                    half_w,
                    half_h,
                    -half_h,
                    0.1f,
                    100.0f,
                    cone.projection
                );
            }
        }

    }

    void render(const graphics::internal::FrameData& fd)
    {
        auto& ctx = graphics::internal::context;

        // The starter's prepare() waits for the previous frame before render().
        // Upload here, after that wait, rather than overwriting GPU data in update().
        for (uint32_t i = 0; i < CONE_COUNT; ++i) {
            memcpy(vk_uniform_buffer_memory[i]->model, g_cones[i].g_model,
                sizeof(g_cones[i].g_model));
            memcpy(vk_uniform_buffer_memory[i]->view, g_view, sizeof(g_view));
            memcpy(
                vk_uniform_buffer_memory[i]->projection,
                g_cones[i].projection,
                sizeof(g_cones[i].projection)
            );
            memcpy(vk_uniform_buffer_memory[i]->tint, g_cones[i].g_tint,
                sizeof(g_cones[i].g_tint));
            if (vmaFlushAllocation(ctx.allocator, vk_uniform_buffer_allocation[i],
                0, sizeof(GlobalUniforms)) != VK_SUCCESS) {
                std::cerr << "Failed to flush uniform buffer\n";
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
        for (uint32_t i = 0; i < CONE_COUNT; ++i) {
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_pipeline_layout, 0, 1, &vk_descriptor_set[i], 0, nullptr);

            vkCmdDrawIndexed(fd.command_buffer, vk_index_count, 1, 0, 0, 0);

        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application