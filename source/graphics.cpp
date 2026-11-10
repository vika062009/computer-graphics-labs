#include "graphics.hpp"
#include "graphics_internal.hpp"
#include "lodepng.h"

#include <cstring>
#include <iostream>
#include <vector>

namespace graphics {

    bool createMappedBufferRaw(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkBuffer& buffer,
        VmaAllocation& allocation,
        void*& memory,
        const void* initial_data)
    {
        auto& ctx = internal::context;

        if (buffer != VK_NULL_HANDLE) {
            std::cerr << "Buffer already exists\n";
            return false;
        }

        memory = nullptr;

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

        if (vmaCreateBuffer(ctx.allocator, &buffer_info, &allocation_info,
            &buffer, &allocation, &memory_info) != VK_SUCCESS) {
            buffer = VK_NULL_HANDLE;
            allocation = nullptr;
            std::cerr << "Failed to create buffer\n";
            return false;
        }

        memory = memory_info.pMappedData;

        if (!memory) {
            std::cerr << "Buffer memory is not mapped\n";
            vmaDestroyBuffer(ctx.allocator, buffer, allocation);
            buffer = VK_NULL_HANDLE;
            allocation = nullptr;
            return false;
        }

        if (initial_data) {
            std::memcpy(memory, initial_data, static_cast<size_t>(size));

            if (vmaFlushAllocation(ctx.allocator, allocation, 0, size)
                != VK_SUCCESS) {
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

    namespace {

        template<class Record>
        bool uploadOnce(VkDeviceSize bytes, const void* data, Record record)
        {
            auto& ctx = internal::context;

            VkBuffer staging{};
            VmaAllocation allocation{};
            void* mapped{};

            if (!createMappedBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                staging, allocation, mapped, data)) {
                return false;
            }

            VkCommandPool pool{};
            VkFence fence{};

            const bool success = [&]() {
                VkCommandPoolCreateInfo pool_info{
                    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO
                };
                pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
                pool_info.queueFamilyIndex = ctx.graphics_queue_index;

                if (vkCreateCommandPool(ctx.device, &pool_info, nullptr, &pool)
                    != VK_SUCCESS) {
                    return false;
                }

                VkCommandBufferAllocateInfo alloc_info{
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
                };
                alloc_info.commandPool = pool;
                alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                alloc_info.commandBufferCount = 1;

                VkCommandBuffer cmd{};
                if (vkAllocateCommandBuffers(ctx.device, &alloc_info, &cmd)
                    != VK_SUCCESS) {
                    return false;
                }

                VkCommandBufferBeginInfo begin_info{
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
                };
                begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

                if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
                    return false;
                }

                record(cmd, staging);

                if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
                    return false;
                }

                VkFenceCreateInfo fence_info{
                    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
                };

                if (vkCreateFence(ctx.device, &fence_info, nullptr, &fence)
                    != VK_SUCCESS) {
                    return false;
                }

                VkSubmitInfo submit_info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cmd;

                if (vkQueueSubmit(ctx.graphics_queue, 1, &submit_info, fence)
                    != VK_SUCCESS) {
                    return false;
                }

                return vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX)
                    == VK_SUCCESS;
                }();

            vkDestroyFence(ctx.device, fence, nullptr);
            vkDestroyCommandPool(ctx.device, pool, nullptr);
            vmaDestroyBuffer(ctx.allocator, staging, allocation);

            return success;
        }

        VmaAllocationCreateInfo gpuMemory()
        {
            VmaAllocationCreateInfo info{};
            info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            return info;
        }

    } 

    bool uploadGeometry(
        VkDeviceSize bytes,
        VkBufferUsageFlags usage,
        const void* data,
        VkBuffer& buffer,
        VmaAllocation& allocation)
    {
        auto& ctx = internal::context;

        VkBufferCreateInfo buffer_info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        buffer_info.size = bytes;
        buffer_info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        const auto allocation_info = gpuMemory();

        if (vmaCreateBuffer(ctx.allocator, &buffer_info, &allocation_info,
            &buffer, &allocation, nullptr) != VK_SUCCESS) {
            return false;
        }

        const bool copied = uploadOnce(bytes, data,
            [&](VkCommandBuffer cmd, VkBuffer staging) {
                VkBufferCopy region{ 0, 0, bytes };
                vkCmdCopyBuffer(cmd, staging, buffer, 1, &region);

                VkBufferMemoryBarrier barrier{
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER
                };
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = (usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
                    ? VK_ACCESS_INDEX_READ_BIT
                    : VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = buffer;
                barrier.offset = 0;
                barrier.size = bytes;

                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                    0, 0, nullptr, 1, &barrier, 0, nullptr
                );
            });

        if (copied) return true;

        vmaDestroyBuffer(ctx.allocator, buffer, allocation);
        buffer = VK_NULL_HANDLE;
        allocation = nullptr;
        return false;
    }

    void destroyTexture(Texture& texture)
    {
        auto& ctx = internal::context;

        vkDestroySampler(ctx.device, texture.sampler, nullptr);
        vkDestroyImageView(ctx.device, texture.view, nullptr);

        if (texture.image) {
            vmaDestroyImage(ctx.allocator, texture.image, texture.allocation);
        }

        texture = {};
    }

    bool loadTexture(const char* path, Texture& texture)
    {
        auto& ctx = internal::context;

        std::vector<unsigned char> pixels;
        unsigned width{}, height{};

        if (const unsigned error = lodepng::decode(pixels, width, height, path)) {
            std::cerr << path << ": " << lodepng_error_text(error) << '\n';
            return false;
        }

        VkImageCreateInfo image_info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = VK_FORMAT_R8G8B8A8_SRGB;
        image_info.extent = { width, height, 1 };
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        const auto allocation_info = gpuMemory();

        if (vmaCreateImage(ctx.allocator, &image_info, &allocation_info,
            &texture.image, &texture.allocation, nullptr)
            != VK_SUCCESS) {
            return false;
        }

        const VkImageSubresourceRange range{
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
        };

        const bool copied = uploadOnce(pixels.size(), pixels.data(),
            [&](VkCommandBuffer cmd, VkBuffer staging) {
                VkImageMemoryBarrier barrier{
                    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER
                };
                barrier.image = texture.image;
                barrier.subresourceRange = range;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &barrier
                );

                VkBufferImageCopy region{};
                region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                region.imageExtent = image_info.extent;

                vkCmdCopyBufferToImage(
                    cmd, staging, texture.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
                );

                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &barrier
                );
            });

        if (!copied) {
            destroyTexture(texture);
            return false;
        }

        VkImageViewCreateInfo view_info{
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO
        };
        view_info.image = texture.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = image_info.format;
        view_info.subresourceRange = range;

        if (vkCreateImageView(ctx.device, &view_info, nullptr, &texture.view)
            != VK_SUCCESS) {
            destroyTexture(texture);
            return false;
        }

        VkSamplerCreateInfo sampler_info{
            VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO
        };
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        if (vkCreateSampler(ctx.device, &sampler_info, nullptr, &texture.sampler)
            != VK_SUCCESS) {
            destroyTexture(texture);
            return false;
        }

        return true;
    }

} // namespace graphics