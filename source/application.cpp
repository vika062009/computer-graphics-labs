#include "application.hpp"
#include "graphics.hpp"

#include <imgui.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {
    namespace {

        constexpr uint32_t CONE_COUNT = 3;
        constexpr uint32_t CONE_SEGMENTS = 50;
        constexpr float CONE_RADIUS = 0.5f;
        constexpr float CONE_HEIGHT = 1.0f;
        constexpr float PI = 3.14159265358979323846f;

        constexpr uint32_t MATERIAL_COUNT = 3;
        constexpr uint32_t BARS = 1;

        constexpr const char* texture_files[MATERIAL_COUNT] = {
            "../textures/lena.png",
            "../textures/bars.png",
            "../textures/elina.png"
        };

        const char* const material_names[MATERIAL_COUNT] = {
            "Lena", "Bars", "Elina"
        };

        struct Vertex {
            float position[3];
            float uv[2];
            float normal[3];
        };

        struct alignas(16) SceneUniforms {
            float view_projection[4][4];
            alignas(16) float camera_position[3];
            alignas(16) float directional_light_direction[3];
            alignas(16) float directional_light_color[3];
            alignas(16) float ambient_light[3];
            uint32_t point_light_count;
            uint32_t spot_light_count;
        };

        struct alignas(16) ModelUniform {
            float model[4][4];
            alignas(16) float albedo[3];
            alignas(16) float specular[3];
            float alpha;
            alignas(16) uint32_t unlit;
        };

        struct alignas(16) PointLight {
            alignas(16) float position[3];
            alignas(16) float color[3];
            float intensity;
        };

        struct alignas(16) SpotLight {
            alignas(16) float position[3];
            alignas(16) float color[3];
            alignas(16) float direction[3];
            float angle;
            float outer_angle;
            float intensity;
        };

        struct TextureEffect {
            uint32_t enabled;
            float time;
        };

        static_assert(sizeof(SceneUniforms) == 144);
        static_assert(offsetof(SceneUniforms, point_light_count) == 124);
        static_assert(offsetof(SceneUniforms, spot_light_count) == 128);
        static_assert(sizeof(ModelUniform) == 112);
        static_assert(offsetof(ModelUniform, alpha) == 92);
        static_assert(offsetof(ModelUniform, unlit) == 96);
        static_assert(sizeof(PointLight) == 32);
        static_assert(sizeof(SpotLight) == 64);
        static_assert(offsetof(SpotLight, angle) == 44);
        static_assert(offsetof(SpotLight, outer_angle) == 48);
        static_assert(sizeof(TextureEffect) == 8);
        static_assert(offsetof(TextureEffect, time) == 4);

        struct Buffer {
            VkBuffer buffer{};
            VmaAllocation allocation{};
            void* memory{};
            VkDeviceSize size{};
        };

        struct CameraState {
            float position[3] = { 0, 0, -3 };
            float rotation_degrees[3]{};
            int projection_mode = 0;
            float fov_degrees = 60;
            float ortho_scale = 1.5f;
            float projection[4][4]{};
        };

        struct ConeState {
            float position[3]{};
            float rotation_degrees[3]{};
            float scale[3] = { 1, 1, 1 };

            bool playing = true;
            float speed = 1;
            float radius = 1.5f;
            float height = 0;
            float rotation_speed = 1;
            float anim_time = 0;

            float albedo[3] = { 1, 1, 1 };
            float specular[3] = { 0.25f, 0.25f, 0.25f };
            float alpha = 32;
            float model[4][4]{};

            int material = 0;
            bool texture_effect = false;
        };

        auto& ctx = graphics::internal::context;

        CameraState camera;
        ConeState cones[CONE_COUNT];
        int selected_cone = 0;
        double last_time = 0;
        float view[4][4]{};

        float light_direction[3] = { -0.5f, 1, 1 };
        float light_color[3] = { 1, 1, 1 };
        float light_intensity = 1;
        float ambient_color[3] = { 1, 1, 1 };
        float ambient_intensity = 0.15f;

        std::vector<PointLight> point_lights;
        std::vector<SpotLight> spot_lights;
        bool lights_dirty = false;
        uint32_t uploaded_points = 0;
        uint32_t uploaded_spots = 0;

        bool show_markers = true;
        bool show_spot_cones = true;
        float marker_size = 0.12f;
        float spot_visual_length = 1.5f;

        Buffer vertex_buffer, index_buffer, marker_buffer;
        Buffer scene_buffer, model_buffer, point_buffer, spot_buffer;
        uint32_t index_count = 0;
        uint32_t cube_vertex_count = 0;
        uint32_t line_vertex_count = 0;
        VkDeviceSize model_stride = 0;
        size_t model_capacity = 0;

        graphics::Texture textures[MATERIAL_COUNT]{};
        graphics::Texture black{}, eyes{}, glow{};

        VkDescriptorSetLayout scene_layout{}, model_layout{};
        VkDescriptorSet scene_set{};
        VkDescriptorSet material_sets[MATERIAL_COUNT]{};
        VkDescriptorPool descriptor_pool{};
        VkPipelineLayout pipeline_layout{};
        VkPipeline pipeline{}, line_pipeline{};
        VkShaderModule vertex_shader{}, fragment_shader{};

        // Матрицы хранятся как [столбец][строка].
        void mat4_identity(float m[4][4])
        {
            std::memset(m, 0, sizeof(float) * 16);
            m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1;
        }

        void mat4_multiply(const float a[4][4], const float b[4][4], float out[4][4])
        {
            float r[4][4]{};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < 4; ++k)
                        r[i][j] += a[k][j] * b[i][k];
            std::memcpy(out, r, sizeof(r));
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

        void mat4_rotation_x(float a, float m[4][4])
        {
            mat4_identity(m);
            const float c = std::cos(a), s = std::sin(a);
            m[1][1] = c; m[2][1] = -s;
            m[1][2] = s; m[2][2] = c;
        }

        void mat4_rotation_y(float a, float m[4][4])
        {
            mat4_identity(m);
            const float c = std::cos(a), s = std::sin(a);
            m[0][0] = c;  m[2][0] = s;
            m[0][2] = -s; m[2][2] = c;
        }

        void mat4_rotation_z(float a, float m[4][4])
        {
            mat4_identity(m);
            const float c = std::cos(a), s = std::sin(a);
            m[0][0] = c; m[1][0] = -s;
            m[0][1] = s; m[1][1] = c;
        }

        float radians(float degrees)
        {
            return degrees * PI / 180.0f;
        }

        void updateCamera()
        {
            float rx[4][4], ry[4][4], rz[4][4], t[4][4], tmp[4][4];

            mat4_rotation_x(-radians(camera.rotation_degrees[0]), rx);
            mat4_rotation_y(-radians(camera.rotation_degrees[1]), ry);
            mat4_rotation_z(-radians(camera.rotation_degrees[2]), rz);
            mat4_translation(-camera.position[0], -camera.position[1], -camera.position[2], t);

            mat4_multiply(rx, ry, tmp);
            mat4_multiply(rz, tmp, tmp);
            mat4_multiply(tmp, t, view);

            const auto extent = ctx.swapchain_extent;
            if (!extent.width || !extent.height) return;

            const float aspect = float(extent.width) / float(extent.height);
            constexpr float n = 0.1f, f = 100.0f;
            auto& p = camera.projection;
            std::memset(p, 0, sizeof(p));

            if (camera.projection_mode == 0) {
                const float h = std::tan(radians(camera.fov_degrees) * 0.5f);
                p[0][0] = 1 / (aspect * h);
                p[1][1] = 1 / h;
                p[2][2] = f / (f - n);
                p[2][3] = 1;
                p[3][2] = -f * n / (f - n);
            }
            else {
                p[0][0] = 1 / (camera.ortho_scale * aspect);
                p[1][1] = 1 / camera.ortho_scale;
                p[2][2] = 1 / (f - n);
                p[3][2] = -n / (f - n);
                p[3][3] = 1;
            }
        }

        void updateCone(ConeState& c, float dt)
        {
            if (c.playing) c.anim_time += dt * c.speed;
            const float a = c.anim_time;
            float t[4][4], s[4][4], rx[4][4], ry[4][4], rz[4][4], tmp[4][4];

            mat4_translation(c.position[0] + c.radius * std::cos(a),
                c.position[1] + c.height,
                c.position[2] + c.radius * std::sin(a), t);
            mat4_rotation_x(radians(c.rotation_degrees[0]), rx);
            mat4_rotation_y(radians(c.rotation_degrees[1]) + a * c.rotation_speed, ry);
            mat4_rotation_z(radians(c.rotation_degrees[2]), rz);
            mat4_scale(c.scale[0], c.scale[1], c.scale[2], s);

            mat4_multiply(ry, rx, tmp);
            mat4_multiply(tmp, rz, tmp);
            mat4_multiply(tmp, s, tmp);
            mat4_multiply(t, tmp, c.model);
        }

        void destroyBuffer(Buffer& b)
        {
            if (b.buffer) vmaDestroyBuffer(ctx.allocator, b.buffer, b.allocation);
            b = {};
        }

        bool createMapped(Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage)
        {
            if (!graphics::createMappedBuffer(size, usage, b.buffer, b.allocation, b.memory))
                return false;
            b.size = size;
            return true;
        }

        bool createConeGeometry()
        {
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;
            const float h = CONE_HEIGHT, r = CONE_RADIUS;
            const float len = std::sqrt(h * h + r * r);

            for (uint32_t i = 0; i <= CONE_SEGMENTS; ++i) {
                const float u = float(i) / CONE_SEGMENTS;
                const float a = 2 * PI * float(i % CONE_SEGMENTS) / CONE_SEGMENTS;
                const float c = std::cos(a), s = std::sin(a);
                vertices.push_back({ {r * c, 0, r * s}, {u, 1}, {h * c / len, -r / len, h * s / len} });
            }

            const uint32_t apex = static_cast<uint32_t>(vertices.size());
            for (uint32_t i = 0; i < CONE_SEGMENTS; ++i)
                vertices.push_back({ {0, -h, 0}, {(float(i) + 0.5f) / CONE_SEGMENTS, 0}, {0,-1,0} });

            const uint32_t center = static_cast<uint32_t>(vertices.size());
            vertices.push_back({ {0,0,0}, {0.5f,0.5f}, {0,1,0} });
            const uint32_t base = center + 1;

            for (uint32_t i = 0; i < CONE_SEGMENTS; ++i) {
                const float a = 2 * PI * float(i) / CONE_SEGMENTS;
                const float c = std::cos(a), s = std::sin(a);
                vertices.push_back({ {r * c,0,r * s}, {0.5f + 0.5f * c,0.5f + 0.5f * s}, {0,1,0} });
                indices.insert(indices.end(), {
                    apex + i, i, i + 1, center, base + (i + 1) % CONE_SEGMENTS, base + i
                    });
            }

            index_count = static_cast<uint32_t>(indices.size());
            return graphics::uploadGeometry(vertices.size() * sizeof(Vertex),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data(),
                vertex_buffer.buffer, vertex_buffer.allocation)
                && graphics::uploadGeometry(indices.size() * sizeof(uint32_t),
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data(),
                    index_buffer.buffer, index_buffer.allocation);
        }

        bool createLightMarkerGeometry()
        {
            std::vector<Vertex> v;
            const float corners[8][3] = {
                {-0.5f,-0.5f,-0.5f}, {0.5f,-0.5f,-0.5f},
                {0.5f,0.5f,-0.5f}, {-0.5f,0.5f,-0.5f},
                {-0.5f,-0.5f,0.5f}, {0.5f,-0.5f,0.5f},
                {0.5f,0.5f,0.5f}, {-0.5f,0.5f,0.5f}
            };
            const unsigned ids[] = {
                0,1,2, 0,2,3, 4,6,5, 4,7,6, 0,4,5, 0,5,1,
                3,2,6, 3,6,7, 0,3,7, 0,7,4, 1,5,6, 1,6,2
            };
            auto add = [&](float x, float y, float z) {
                v.push_back({ {x,y,z}, {0,0}, {0,0,0} });
                };
            for (auto i : ids) add(corners[i][0], corners[i][1], corners[i][2]);
            cube_vertex_count = static_cast<uint32_t>(v.size());

            for (int i = 0; i < 32; ++i) {
                float a = 2 * PI * i / 32, b = 2 * PI * (i + 1) / 32;
                add(std::cos(a), std::sin(a), 1);
                add(std::cos(b), std::sin(b), 1);
                if (i % 8 == 0) {
                    add(0, 0, 0);
                    add(std::cos(a), std::sin(a), 1);
                }
            }
            add(0, 0, 0);
            add(0, 0, 1);
            line_vertex_count = static_cast<uint32_t>(v.size()) - cube_vertex_count;

            return graphics::uploadGeometry(v.size() * sizeof(Vertex),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, v.data(),
                marker_buffer.buffer, marker_buffer.allocation);
        }

        bool makeSpotMarkerMatrix(const SpotLight& light, float m[4][4])
        {
            float f[3], right[3], up[3], ref[3] = { 0,1,0 };
            const auto& d = light.direction;
            const float len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            if (len < 0.0001f) return false;

            for (int i = 0; i < 3; ++i) f[i] = d[i] / len;
            if (std::fabs(f[1]) > 0.99f) { ref[0] = 1; ref[1] = 0; }

            auto cross = [](const float* a, const float* b, float* out) {
                out[0] = a[1] * b[2] - a[2] * b[1];
                out[1] = a[2] * b[0] - a[0] * b[2];
                out[2] = a[0] * b[1] - a[1] * b[0];
                };
            cross(ref, f, right);
            const float k = std::sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
            for (float& x : right) x /= k;
            cross(f, right, up);

            const float c = std::fmax(0.0001f, std::fmin(1.0f, light.outer_angle));
            const float radius = spot_visual_length * std::sqrt(1 - c * c) / c;
            mat4_identity(m);
            for (int i = 0; i < 3; ++i) {
                m[0][i] = right[i] * radius;
                m[1][i] = up[i] * radius;
                m[2][i] = f[i] * spot_visual_length;
                m[3][i] = light.position[i];
            }
            return true;
        }

        void writeDescriptor(VkDescriptorSet set, uint32_t binding,
            VkDescriptorType type, const Buffer& buffer, VkDeviceSize range)
        {
            VkDescriptorBufferInfo info{ buffer.buffer, 0, range };
            VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w.dstSet = set;
            w.dstBinding = binding;
            w.descriptorCount = 1;
            w.descriptorType = type;
            w.pBufferInfo = &info;
            vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
        }

        bool ensureModelCapacity(size_t required)
        {
            if (required <= model_capacity) return true;
            if (required > UINT32_MAX / model_stride) return false;

            Buffer next{};
            if (!createMapped(next, required * model_stride, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT))
                return false;

            Buffer old = model_buffer;
            model_buffer = next;
            model_capacity = required;
            for (auto set : material_sets)
                if (set) writeDescriptor(set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    model_buffer, sizeof(ModelUniform));
            destroyBuffer(old);
            return true;
        }

        bool createUniformBuffer()
        {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(ctx.physical_device, &props);
            const auto a = props.limits.minUniformBufferOffsetAlignment;
            model_stride = (sizeof(ModelUniform) + a - 1) / a * a;
            return createMapped(scene_buffer, sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
                && ensureModelCapacity(CONE_COUNT);
        }

        void writeModel(size_t slot, const ModelUniform& data)
        {
            std::memcpy(static_cast<unsigned char*>(model_buffer.memory) + slot * model_stride,
                &data, sizeof(data));
        }

        void bindModel(VkCommandBuffer cmd, size_t slot, int material)
        {
            const uint32_t offset = static_cast<uint32_t>(slot * model_stride);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                1, 1, &material_sets[material], 1, &offset);
        }

        template<class T>
        bool makeLightBuffer(const std::vector<T>& list, Buffer& out)
        {
            const size_t count = list.empty() ? 1 : list.size();
            if (!createMapped(out, count * sizeof(T), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;

            auto* dst = static_cast<unsigned char*>(out.memory);
            std::memset(dst, 0, static_cast<size_t>(out.size));
            for (size_t i = 0; i < list.size(); ++i) {
                T data = list[i];
                for (float& c : data.color) c *= data.intensity;
                std::memcpy(dst + i * sizeof(T), &data, sizeof(T));
            }
            if (vmaFlushAllocation(ctx.allocator, out.allocation, 0, out.size) != VK_SUCCESS) {
                destroyBuffer(out);
                return false;
            }
            return true;
        }

        bool updateLightStorageBuffers()
        {
            if (!lights_dirty) return true;
            Buffer points{}, spots{};
            if (!makeLightBuffer(point_lights, points)) return false;
            if (!makeLightBuffer(spot_lights, spots)) {
                destroyBuffer(points);
                return false;
            }
            if (vkQueueWaitIdle(ctx.graphics_queue) != VK_SUCCESS) {
                destroyBuffer(points);
                destroyBuffer(spots);
                return false;
            }

            writeDescriptor(scene_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, points, points.size);
            writeDescriptor(scene_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spots, spots.size);
            destroyBuffer(point_buffer);
            destroyBuffer(spot_buffer);
            point_buffer = points;
            spot_buffer = spots;
            uploaded_points = static_cast<uint32_t>(point_lights.size());
            uploaded_spots = static_cast<uint32_t>(spot_lights.size());
            lights_dirty = false;
            return true;
        }

        bool loadMaterialTextures()
        {
            for (uint32_t i = 0; i < MATERIAL_COUNT; ++i) {
                if (!graphics::loadTexture(texture_files[i], textures[i])) return false;
            }
            return graphics::loadTexture("../textures/black.png", black)
                && graphics::loadTexture("../textures/bars_specular.png", eyes)
                && graphics::loadTexture("../textures/bars_emissive.png", glow);
        }

        bool createDescriptorSet()
        {
            constexpr VkShaderStageFlags both = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            const VkDescriptorSetLayoutBinding scene_bindings[] = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, both, nullptr},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
            };
            const VkDescriptorSetLayoutBinding model_bindings[] = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, both, nullptr},
                {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
            };

            auto makeLayout = [&](const VkDescriptorSetLayoutBinding* bindings, uint32_t count,
                VkDescriptorSetLayout& layout) {
                    VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
                    info.bindingCount = count;
                    info.pBindings = bindings;
                    return vkCreateDescriptorSetLayout(ctx.device, &info, nullptr, &layout) == VK_SUCCESS;
                };
            if (!makeLayout(scene_bindings, 3, scene_layout) ||
                !makeLayout(model_bindings, 4, model_layout)) return false;

            const VkDescriptorPoolSize sizes[] = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, MATERIAL_COUNT},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * MATERIAL_COUNT}
            };
            VkDescriptorPoolCreateInfo pool{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            pool.maxSets = 1 + MATERIAL_COUNT;
            pool.poolSizeCount = 4;
            pool.pPoolSizes = sizes;
            if (vkCreateDescriptorPool(ctx.device, &pool, nullptr, &descriptor_pool) != VK_SUCCESS)
                return false;

            auto alloc = [&](VkDescriptorSetLayout layout, VkDescriptorSet& set) {
                VkDescriptorSetAllocateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
                info.descriptorPool = descriptor_pool;
                info.descriptorSetCount = 1;
                info.pSetLayouts = &layout;
                return vkAllocateDescriptorSets(ctx.device, &info, &set) == VK_SUCCESS;
                };
            if (!alloc(scene_layout, scene_set)) return false;
            writeDescriptor(scene_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                scene_buffer, sizeof(SceneUniforms));
            writeDescriptor(scene_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, point_buffer, point_buffer.size);
            writeDescriptor(scene_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spot_buffer, spot_buffer.size);

            for (uint32_t i = 0; i < MATERIAL_COUNT; ++i) {
                auto& set = material_sets[i];
                if (!alloc(model_layout, set)) return false;
                writeDescriptor(set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    model_buffer, sizeof(ModelUniform));

                // binding 1 — фотография, 2 — карта блика, 3 — карта свечения.
                const graphics::Texture* maps[] = {
                    &textures[i], i == BARS ? &eyes : &black, i == BARS ? &glow : &black
                };
                for (uint32_t k = 0; k < 3; ++k) {
                    VkDescriptorImageInfo image{
                        maps[k]->sampler, maps[k]->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    };
                    VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                    w.dstSet = set;
                    w.dstBinding = 1 + k;
                    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    w.descriptorCount = 1;
                    w.pImageInfo = &image;
                    vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
                }
            }
            return true;
        }

        VkShaderModule loadShaderModule(const char* path)
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file) {
                std::cerr << "Failed to open shader: " << path << '\n';
                return VK_NULL_HANDLE;
            }
            const std::streamoff size = file.tellg();
            if (size <= 0 || size % 4 != 0) return VK_NULL_HANDLE;
            std::vector<uint32_t> code(static_cast<size_t>(size) / 4);
            file.seekg(0);
            if (!file.read(reinterpret_cast<char*>(code.data()), size)) return VK_NULL_HANDLE;

            VkShaderModuleCreateInfo info{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            info.codeSize = static_cast<size_t>(size);
            info.pCode = code.data();
            VkShaderModule module{};
            if (vkCreateShaderModule(ctx.device, &info, nullptr, &module) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            return module;
        }

        bool createPipeline()
        {
            vertex_shader = loadShaderModule("../shaders/cone.vert.spv");
            fragment_shader = loadShaderModule("../shaders/cone.frag.spv");
            if (!vertex_shader || !fragment_shader) return false;

            VkPipelineShaderStageCreateInfo stages[2]{};
            for (auto& stage : stages) {
                stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                stage.pName = "main";
            }
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertex_shader;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragment_shader;

            const VkVertexInputBindingDescription binding{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
            const VkVertexInputAttributeDescription attrs[] = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
                {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
                {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)}
            };
            VkPipelineVertexInputStateCreateInfo input{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
            input.vertexBindingDescriptionCount = 1;
            input.pVertexBindingDescriptions = &binding;
            input.vertexAttributeDescriptionCount = 3;
            input.pVertexAttributeDescriptions = attrs;

            VkPipelineInputAssemblyStateCreateInfo assembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewport{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
            viewport.viewportCount = viewport.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1;

            VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depth{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
            depth.depthTestEnable = VK_TRUE;
            depth.depthWriteEnable = VK_TRUE;
            depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineColorBlendAttachmentState attachment{};
            attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo blend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
            blend.attachmentCount = 1;
            blend.pAttachments = &attachment;

            const VkDynamicState states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynamic{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
            dynamic.dynamicStateCount = 2;
            dynamic.pDynamicStates = states;

            const VkDescriptorSetLayout layouts[] = { scene_layout, model_layout };
            const VkPushConstantRange range{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(TextureEffect) };
            VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            li.setLayoutCount = 2;
            li.pSetLayouts = layouts;
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &range;
            if (vkCreatePipelineLayout(ctx.device, &li, nullptr, &pipeline_layout) != VK_SUCCESS)
                return false;

            VkGraphicsPipelineCreateInfo info{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
            info.stageCount = 2;
            info.pStages = stages;
            info.pVertexInputState = &input;
            info.pInputAssemblyState = &assembly;
            info.pViewportState = &viewport;
            info.pRasterizationState = &raster;
            info.pMultisampleState = &ms;
            info.pDepthStencilState = &depth;
            info.pColorBlendState = &blend;
            info.pDynamicState = &dynamic;
            info.layout = pipeline_layout;
            info.renderPass = ctx.render_pass;
            info.subpass = 0;

            if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline)
                != VK_SUCCESS) return false;

            assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            depth.depthWriteEnable = VK_FALSE;
            return vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr,
                &line_pipeline) == VK_SUCCESS;
        }

        void drawLightControls()
        {
            auto common = [](auto& light) {
                if (ImGui::DragFloat3("Position", light.position, 0.02f)) lights_dirty = true;
                if (ImGui::ColorEdit3("Color", light.color)) lights_dirty = true;
                if (ImGui::SliderFloat("Intensity", &light.intensity, 0, 30, "%.3f",
                    ImGuiSliderFlags_AlwaysClamp)) lights_dirty = true;
                };

            if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID("point_lights");
                if (ImGui::Button("Add point light")) {
                    point_lights.push_back({ {1.5f,-0.5f,-2}, {1,1,1}, 4 });
                    lights_dirty = true;
                }
                ImGui::Text("Count: %d", static_cast<int>(point_lights.size()));
                for (size_t i = 0; i < point_lights.size();) {
                    ImGui::PushID(static_cast<int>(i));
                    ImGui::Separator();
                    ImGui::Text("Point light %d", static_cast<int>(i + 1));
                    if (ImGui::Button("Delete")) {
                        point_lights.erase(point_lights.begin() + static_cast<std::ptrdiff_t>(i));
                        lights_dirty = true;
                        ImGui::PopID();
                        continue;
                    }
                    common(point_lights[i]);
                    ImGui::PopID();
                    ++i;
                }
                ImGui::PopID();
            }

            if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID("spot_lights");
                if (ImGui::Button("Add spot light")) {
                    spot_lights.push_back({ {1.5f,-0.5f,-2}, {1,1,1}, {0,0,1},
                                           std::cos(radians(15)), std::cos(radians(25)), 4 });
                    lights_dirty = true;
                }
                ImGui::Text("Count: %d", static_cast<int>(spot_lights.size()));
                for (size_t i = 0; i < spot_lights.size();) {
                    ImGui::PushID(static_cast<int>(i));
                    ImGui::Separator();
                    ImGui::Text("Spot light %d", static_cast<int>(i + 1));
                    if (ImGui::Button("Delete")) {
                        spot_lights.erase(spot_lights.begin() + static_cast<std::ptrdiff_t>(i));
                        lights_dirty = true;
                        ImGui::PopID();
                        continue;
                    }
                    auto& light = spot_lights[i];
                    common(light);
                    if (ImGui::DragFloat3("Ray direction", light.direction, 0.02f)) lights_dirty = true;

                    const auto& d = light.direction;
                    if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < 0.00000001f)
                        ImGui::TextUnformatted("Set a non-zero ray direction.");

                    float inner = std::acos(std::fmax(-1.f, std::fmin(1.f, light.angle))) * 180 / PI;
                    float outer = std::acos(std::fmax(-1.f, std::fmin(1.f, light.outer_angle))) * 180 / PI;
                    bool changed = false;
                    if (ImGui::SliderFloat("Inner half-angle (deg)", &inner, 0, 89, "%.1f",
                        ImGuiSliderFlags_AlwaysClamp)) {
                        if (outer < inner) outer = inner;
                        changed = true;
                    }
                    if (ImGui::SliderFloat("Outer half-angle (deg)", &outer, 0, 89, "%.1f",
                        ImGuiSliderFlags_AlwaysClamp)) {
                        if (inner > outer) inner = outer;
                        changed = true;
                    }
                    if (changed) {
                        light.angle = std::cos(radians(inner));
                        light.outer_angle = std::cos(radians(outer));
                        lights_dirty = true;
                    }
                    ImGui::PopID();
                    ++i;
                }
                ImGui::PopID();
            }
        }

        void drawLightMarkers(VkCommandBuffer cmd)
        {
            if (!show_markers && !show_spot_cones) return;
            const VkDeviceSize zero = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &marker_buffer.buffer, &zero);

            size_t slot = CONE_COUNT;
            const TextureEffect disabled{};
            vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(disabled), &disabled);

            auto draw = [&](const float m[4][4], const float rgb[3], uint32_t count, uint32_t first) {
                ModelUniform data{};
                std::memcpy(data.model, m, sizeof(data.model));
                std::memcpy(data.albedo, rgb, sizeof(data.albedo));
                data.unlit = 1;
                writeModel(slot, data);
                bindModel(cmd, slot++, 0);
                vkCmdDraw(cmd, count, 1, first, 0);
                };

            if (show_markers) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                auto cube = [&](const auto& light) {
                    float m[4][4];
                    mat4_scale(marker_size, marker_size, marker_size, m);
                    for (int i = 0; i < 3; ++i) m[3][i] = light.position[i];
                    draw(m, light.color, cube_vertex_count, 0);
                    };
                for (const auto& light : point_lights) cube(light);
                for (const auto& light : spot_lights) cube(light);
            }
            if (show_spot_cones) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
                const float white[3] = { 1,1,1 };
                for (const auto& light : spot_lights) {
                    float m[4][4];
                    if (makeSpotMarkerMatrix(light, m))
                        draw(m, white, line_vertex_count, cube_vertex_count);
                }
            }
        }

    } // namespace

    bool initialize()
    {
        if (!createConeGeometry() || !createLightMarkerGeometry() ||
            !createUniformBuffer() || !loadMaterialTextures() ||
            !makeLightBuffer(point_lights, point_buffer) ||
            !makeLightBuffer(spot_lights, spot_buffer) ||
            !createDescriptorSet() || !createPipeline()) {
            std::cerr << "Failed to initialize Lab 3\n";
            shutdown();
            return false;
        }

        for (auto& cone : cones) mat4_identity(cone.model);
        cones[0].material = 0;
        cones[0].texture_effect = true;

        cones[1].material = 1;
        cones[1].position[0] = -1.25f;
        cones[1].albedo[1] = 0.4f;
        cones[1].albedo[2] = 0.4f;
        cones[1].playing = false;

        cones[2].material = 2;
        cones[2].position[2] = 0.75f;
        cones[2].albedo[0] = 0.4f;
        cones[2].albedo[1] = 0.6f;
        cones[2].playing = false;

        updateCamera();
        last_time = 0;
        return true;
    }

    void shutdown()
    {
        vkQueueWaitIdle(ctx.graphics_queue);
        vkDestroyPipeline(ctx.device, line_pipeline, nullptr);
        vkDestroyPipeline(ctx.device, pipeline, nullptr);
        vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
        vkDestroyShaderModule(ctx.device, vertex_shader, nullptr);
        vkDestroyShaderModule(ctx.device, fragment_shader, nullptr);
        vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, model_layout, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, scene_layout, nullptr);

        for (auto& tex : textures) graphics::destroyTexture(tex);
        graphics::destroyTexture(black);
        graphics::destroyTexture(eyes);
        graphics::destroyTexture(glow);
        for (auto& set : material_sets) set = VK_NULL_HANDLE;

        destroyBuffer(model_buffer);
        destroyBuffer(scene_buffer);
        destroyBuffer(point_buffer);
        destroyBuffer(spot_buffer);
        destroyBuffer(marker_buffer);
        destroyBuffer(index_buffer);
        destroyBuffer(vertex_buffer);
        model_capacity = 0;
    }

    void update(double time)
    {
        const float dt = last_time == 0 ? 0.f : static_cast<float>(time - last_time);
        last_time = time;

        ImGui::Begin("Lab 3 - Texture mapping");

        ImGui::SeparatorText("Directional light");
        ImGui::DragFloat3("Light ray direction", light_direction, 0.02f);
        ImGui::ColorEdit3("Light color", light_color);
        ImGui::SliderFloat("Light intensity", &light_intensity, 0, 5);

        ImGui::SeparatorText("Ambient light");
        ImGui::ColorEdit3("Ambient color", ambient_color);
        ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0, 1);
        drawLightControls();

        ImGui::SeparatorText("Light visualization");
        ImGui::Checkbox("Show light markers", &show_markers);
        ImGui::Checkbox("Show spot cones", &show_spot_cones);
        ImGui::SliderFloat("Marker size", &marker_size, 0.03f, 0.3f);
        ImGui::SliderFloat("Cone display length", &spot_visual_length, 0.1f, 5);

        ImGui::TextUnformatted("Select cone:");
        ImGui::RadioButton("Cone 1", &selected_cone, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Cone 2", &selected_cone, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Cone 3", &selected_cone, 2);
        auto& c = cones[selected_cone];

        ImGui::SeparatorText("Camera");
        ImGui::SliderFloat3("Camera position", camera.position, -10, 10);
        ImGui::SliderFloat3("Camera rotation (deg)", camera.rotation_degrees, -180, 180);
        ImGui::RadioButton("Perspective", &camera.projection_mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &camera.projection_mode, 1);
        if (camera.projection_mode == 0)
            ImGui::SliderFloat("Camera FOV (deg)", &camera.fov_degrees, 20, 120);
        else
            ImGui::SliderFloat("Camera ortho scale", &camera.ortho_scale, 0.2f, 5);
        if (ImGui::Button("Reset camera")) camera = CameraState{};

        ImGui::SeparatorText("Transform");
        ImGui::SliderFloat3("Position", c.position, -3, 3);
        ImGui::SliderFloat3("Rotation (deg)", c.rotation_degrees, -180, 180);
        ImGui::SliderFloat3("Scale", c.scale, 0.1f, 3, "%.3f", ImGuiSliderFlags_AlwaysClamp);

        ImGui::SeparatorText("Animation");
        ImGui::Checkbox("Playing", &c.playing);
        ImGui::SliderFloat("Speed", &c.speed, 0, 5);
        ImGui::SliderFloat("Trajectory radius", &c.radius, 0, 5);
        ImGui::SliderFloat("Trajectory height", &c.height, -2, 2);
        ImGui::SliderFloat("Self-rotation speed", &c.rotation_speed, 0, 5);
        if (ImGui::Button("Reset animation time")) c.anim_time = 0;

        ImGui::SeparatorText("Material");
        ImGui::Combo("Texture", &c.material, material_names, static_cast<int>(MATERIAL_COUNT));
        ImGui::Checkbox("Animated texture", &c.texture_effect);
        ImGui::ColorEdit3("Albedo multiplier", c.albedo);
        ImGui::ColorEdit3("Specular color", c.specular);
        ImGui::SliderFloat("Alpha (shininess)", &c.alpha, 1, 256);

        ImGui::End();
        for (auto& cone : cones) updateCone(cone, dt);
        updateCamera();
    }

    void render(const graphics::internal::FrameData& fd)
    {
        if (!updateLightStorageBuffers())
            std::cerr << "Failed to update light buffers\n";

        SceneUniforms scene{};
        mat4_multiply(camera.projection, view, scene.view_projection);
        for (int i = 0; i < 3; ++i) {
            scene.camera_position[i] = camera.position[i];
            scene.directional_light_direction[i] = light_direction[i];
            scene.directional_light_color[i] = light_color[i] * light_intensity;
            scene.ambient_light[i] = ambient_color[i] * ambient_intensity;
        }
        scene.point_light_count = uploaded_points;
        scene.spot_light_count = uploaded_spots;
        std::memcpy(scene_buffer.memory, &scene, sizeof(scene));
        if (vmaFlushAllocation(ctx.allocator, scene_buffer.allocation, 0, sizeof(scene)) != VK_SUCCESS)
            std::cerr << "Failed to flush scene UBO\n";

        const size_t needed = CONE_COUNT
            + (show_markers ? point_lights.size() + spot_lights.size() : 0)
            + (show_spot_cones ? spot_lights.size() : 0);
        const bool markers_fit = ensureModelCapacity(needed);

        for (uint32_t i = 0; i < CONE_COUNT; ++i) {
            ModelUniform m{};
            std::memcpy(m.model, cones[i].model, sizeof(m.model));
            std::memcpy(m.albedo, cones[i].albedo, sizeof(m.albedo));
            std::memcpy(m.specular, cones[i].specular, sizeof(m.specular));
            m.alpha = cones[i].alpha;
            writeModel(i, m);
        }

        const auto cmd = fd.command_buffer;
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin);

        VkClearValue clear[2]{};
        clear[0].color.float32[0] = 0.1f;
        clear[0].color.float32[1] = 0.1f;
        clear[0].color.float32[2] = 0.1f;
        clear[0].color.float32[3] = 1;
        clear[1].depthStencil.depth = 1;

        VkRenderPassBeginInfo rp{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        rp.renderPass = ctx.render_pass;
        rp.framebuffer = fd.framebuffer;
        rp.renderArea.extent = ctx.swapchain_extent;
        rp.clearValueCount = 2;
        rp.pClearValues = clear;
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        const VkViewport vp{ 0, 0, float(ctx.swapchain_extent.width),
                                  float(ctx.swapchain_extent.height), 0, 1 };
        const VkRect2D scissor{ {0,0}, ctx.swapchain_extent };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        const VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &zero);
        vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 0, 1, &scene_set, 0, nullptr);

        for (uint32_t i = 0; i < CONE_COUNT; ++i) {
            bindModel(cmd, i, cones[i].material);
            const TextureEffect effect{ cones[i].texture_effect ? 1u : 0u,
                                       static_cast<float>(last_time) };
            vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(effect), &effect);
            vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
        }

        if (markers_fit) drawLightMarkers(cmd);

        if (vmaFlushAllocation(ctx.allocator, model_buffer.allocation, 0,
            (markers_fit ? needed : CONE_COUNT) * model_stride) != VK_SUCCESS)
            std::cerr << "Failed to flush model UBO\n";

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace application