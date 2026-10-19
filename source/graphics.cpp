#include "graphics.hpp"
#include "graphics_internal.hpp"

#include <cstring>
#include <iostream>

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

        if (vmaCreateBuffer(
            ctx.allocator, &buffer_info, &allocation_info,
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

} // namespace graphics