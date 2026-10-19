#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace graphics {

    bool createMappedBufferRaw(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkBuffer& buffer,
        VmaAllocation& allocation,
        void*& memory,
        const void* initial_data);

    template<typename T>
    bool createMappedBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkBuffer& buffer,
        VmaAllocation& allocation,
        T*& memory,
        const void* initial_data = nullptr)
    {
        if (buffer != VK_NULL_HANDLE) {
            return false;
        }

        void* mapped_memory = nullptr;

        const bool success = createMappedBufferRaw(
            size, usage, buffer, allocation, mapped_memory, initial_data);

        memory = static_cast<T*>(mapped_memory);
        return success;
    }

} // namespace graphics