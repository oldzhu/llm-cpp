#include "backend/vulkan/vulkan_backend.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

#ifdef BUILD_VULKAN
#define VK_NO_PROTOTYPES
#include <vulkan.h>

// Dynamic loader: we link vulkan-1.lib at build time
// On Windows, the loader is in the Vulkan SDK
#if defined(_WIN32)
#include <windows.h>
static HMODULE g_vulkan_dll = nullptr;

static PFN_vkGetInstanceProcAddr g_vkGetInstanceProcAddr = nullptr;

#define VK_LOAD(fn) reinterpret_cast<PFN_##fn>(g_vkGetInstanceProcAddr(nullptr, #fn))
#define VK_LOAD_INSTANCE(inst, fn) reinterpret_cast<PFN_##fn>(g_vkGetInstanceProcAddr(static_cast<VkInstance>(inst), #fn))
#else
#include <dlfcn.h>
static void* g_vulkan_so = nullptr;
#endif
#endif // BUILD_VULKAN

#include "backend/cpu_backend.h"  // fallback for backward pass

namespace backend {

// =========================================================================
// Vulkan backend helper: read SPIR-V from file
// =========================================================================
#ifdef BUILD_VULKAN

static std::vector<std::uint32_t> read_spirv_file(const char* path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error(std::string("Vulkan: cannot open SPIR-V file: ") + path);
  const std::size_t size = static_cast<std::size_t>(in.tellg());
  if (size % 4 != 0) throw std::runtime_error("Vulkan: SPIR-V file size not multiple of 4");
  in.seekg(0);
  std::vector<std::uint32_t> code(size / 4);
  in.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));
  return code;
}

// =========================================================================
// Vulkan backend: full Vulkan compute pipeline (teaching-first, explicit)
// =========================================================================

struct VulkanBackend::Impl {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t queue_family = 0;
  VkCommandPool cmd_pool = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkDescriptorPool desc_pool = VK_NULL_HANDLE;

  // Vulkan function pointers (dynamically loaded)
  PFN_vkCreateInstance vkCreateInstance = nullptr;
  PFN_vkDestroyInstance vkDestroyInstance = nullptr;
  PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
  PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
  PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
  PFN_vkCreateDevice vkCreateDevice = nullptr;
  PFN_vkDestroyDevice vkDestroyDevice = nullptr;
  PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
  PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
  PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
  PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
  PFN_vkFreeCommandBuffers vkFreeCommandBuffers = nullptr;
  PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
  PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
  PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
  PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
  PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
  PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
  PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
  PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
  PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
  PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
  PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
  PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
  PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
  PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;
  PFN_vkCreateBuffer vkCreateBuffer = nullptr;
  PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
  PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
  PFN_vkAllocateMemory vkAllocateMemory = nullptr;
  PFN_vkFreeMemory vkFreeMemory = nullptr;
  PFN_vkMapMemory vkMapMemory = nullptr;
  PFN_vkUnmapMemory vkUnmapMemory = nullptr;
  PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
  PFN_vkQueueSubmit vkQueueSubmit = nullptr;
  PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;
  PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
  PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
  PFN_vkCmdDispatch vkCmdDispatch = nullptr;
  PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
  PFN_vkResetCommandBuffer vkResetCommandBuffer = nullptr;

  bool initialise() {
#if defined(_WIN32)
    g_vulkan_dll = LoadLibraryA("vulkan-1.dll");
    if (!g_vulkan_dll) return false;
    g_vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetProcAddress(g_vulkan_dll, "vkGetInstanceProcAddr"));
    if (!g_vkGetInstanceProcAddr) return false;
#else
    g_vulkan_so = dlopen("libvulkan.so", RTLD_NOW);
    if (!g_vulkan_so) return false;
    g_vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        dlsym(g_vulkan_so, "vkGetInstanceProcAddr"));
    if (!g_vkGetInstanceProcAddr) return false;
#endif

#define LOAD(fn) fn = reinterpret_cast<PFN_##fn>(g_vkGetInstanceProcAddr(nullptr, #fn)); if (!fn) return false
    LOAD(vkCreateInstance);
    LOAD(vkDestroyInstance);
    LOAD(vkEnumeratePhysicalDevices);
    LOAD(vkGetPhysicalDeviceProperties);
    LOAD(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD(vkCreateDevice);
    LOAD(vkDestroyDevice);
    LOAD(vkGetDeviceQueue);
    LOAD(vkCreateCommandPool);
    LOAD(vkDestroyCommandPool);
    LOAD(vkAllocateCommandBuffers);
    LOAD(vkFreeCommandBuffers);
    LOAD(vkBeginCommandBuffer);
    LOAD(vkEndCommandBuffer);
    LOAD(vkCreateShaderModule);
    LOAD(vkDestroyShaderModule);
    LOAD(vkCreateComputePipelines);
    LOAD(vkDestroyPipeline);
    LOAD(vkCreatePipelineLayout);
    LOAD(vkDestroyPipelineLayout);
    LOAD(vkCreateDescriptorSetLayout);
    LOAD(vkDestroyDescriptorSetLayout);
    LOAD(vkCreateDescriptorPool);
    LOAD(vkDestroyDescriptorPool);
    LOAD(vkAllocateDescriptorSets);
    LOAD(vkUpdateDescriptorSets);
    LOAD(vkCreateBuffer);
    LOAD(vkDestroyBuffer);
    LOAD(vkGetBufferMemoryRequirements);
    LOAD(vkAllocateMemory);
    LOAD(vkFreeMemory);
    LOAD(vkMapMemory);
    LOAD(vkUnmapMemory);
    LOAD(vkBindBufferMemory);
    LOAD(vkQueueSubmit);
    LOAD(vkQueueWaitIdle);
    LOAD(vkCmdBindPipeline);
    LOAD(vkCmdBindDescriptorSets);
    LOAD(vkCmdDispatch);
    LOAD(vkCmdPipelineBarrier);
    LOAD(vkResetCommandBuffer);
#undef LOAD

    // Create Vulkan instance
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_1;

    const char* layers[] = { };
    const char* extensions[] = { };

    VkInstanceCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app_info;
    ci.enabledLayerCount = 0;
    ci.enabledExtensionCount = 0;

    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) return false;

    // Pick first GPU
    std::uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr);
    if (gpu_count == 0) return false;
    std::vector<VkPhysicalDevice> gpus(gpu_count);
    vkEnumeratePhysicalDevices(instance, &gpu_count, gpus.data());
    physical = gpus[0];

    // Find compute queue
    std::uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qf_count, qf_props.data());
    queue_family = static_cast<std::uint32_t>(-1);
    for (std::uint32_t i = 0; i < qf_count; ++i) {
      if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        queue_family = i;
        break;
      }
    }
    if (queue_family == static_cast<std::uint32_t>(-1)) return false;

    // Create device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    if (vkCreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS) return false;
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    // Create command pool
    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = queue_family;
    if (vkCreateCommandPool(device, &cpci, nullptr, &cmd_pool) != VK_SUCCESS) return false;

    // Descriptor set layout: 3 storage buffers (A, B, C)
    VkDescriptorSetLayoutBinding bindings[3] = {};
    for (int i = 0; i < 3; ++i) {
      bindings[i].binding = static_cast<std::uint32_t>(i);
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl) != VK_SUCCESS) return false;

    // Pipeline layout: push constants for M, K, N
    VkPushConstantRange pc_range = {};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = 3 * sizeof(std::int32_t);

    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc_range;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &pipe_layout) != VK_SUCCESS) return false;

    // Descriptor pool
    VkDescriptorPoolSize dp_size = {};
    dp_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dp_size.descriptorCount = 3;

    VkDescriptorPoolCreateInfo dpci = {};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dp_size;
    if (vkCreateDescriptorPool(device, &dpci, nullptr, &desc_pool) != VK_SUCCESS) return false;

    // Load SPIR-V shader
    std::vector<std::uint32_t> spirv;
    try {
      spirv = read_spirv_file("shaders/matmul_fwd.spv");
    } catch (...) {
      return false; // SPIR-V not found — user needs to compile the shader
    }

    VkShaderModuleCreateInfo smci = {};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size() * sizeof(std::uint32_t);
    smci.pCode = spirv.data();

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device, &smci, nullptr, &shader_module) != VK_SUCCESS) return false;

    // Compute pipeline
    VkComputePipelineCreateInfo cpci2 = {};
    cpci2.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci2.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci2.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci2.stage.module = shader_module;
    cpci2.stage.pName = "main";
    cpci2.layout = pipe_layout;

    VkResult result = vkCreateComputePipelines(device, nullptr, 1, &cpci2, nullptr, &pipeline);
    vkDestroyShaderModule(device, shader_module, nullptr);

    if (result != VK_SUCCESS) return false;

    return true;
  }

  // Helper: create a buffer + allocate + bind device memory
  bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* buf, VkDeviceMemory* mem) {
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bci, nullptr, buf) != VK_SUCCESS) return false;

    VkMemoryRequirements mreq;
    vkGetBufferMemoryRequirements(device, *buf, &mreq);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physical, &mp);

    std::uint32_t mem_type = static_cast<std::uint32_t>(-1);
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
      if ((mreq.memoryTypeBits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & want) == want)) {
        mem_type = i;
        break;
      }
    }
    if (mem_type == static_cast<std::uint32_t>(-1)) return false;

    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mreq.size;
    mai.memoryTypeIndex = mem_type;
    if (vkAllocateMemory(device, &mai, nullptr, mem) != VK_SUCCESS) return false;
    vkBindBufferMemory(device, *buf, *mem, 0);
    return true;
  }

  void run_matmul_fwd(int M, int K, int N, const float* a, const float* b, float* c) {
    const std::size_t size_a = static_cast<std::size_t>(M) * K * sizeof(float);
    const std::size_t size_b = static_cast<std::size_t>(K) * N * sizeof(float);
    const std::size_t size_c = static_cast<std::size_t>(M) * N * sizeof(float);

    // Create buffers
    VkBuffer buf_a, buf_b, buf_c;
    VkDeviceMemory mem_a, mem_b, mem_c;

    if (!create_buffer(size_a, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buf_a, &mem_a))
      throw std::runtime_error("Vulkan: failed to create buffer A");
    if (!create_buffer(size_b, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buf_b, &mem_b))
      throw std::runtime_error("Vulkan: failed to create buffer B");
    if (!create_buffer(size_c, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &buf_c, &mem_c))
      throw std::runtime_error("Vulkan: failed to create buffer C");

    // Upload A and B
    void* mapped = nullptr;
    vkMapMemory(device, mem_a, 0, size_a, 0, &mapped);
    std::memcpy(mapped, a, size_a);
    vkUnmapMemory(device, mem_a);

    vkMapMemory(device, mem_b, 0, size_b, 0, &mapped);
    std::memcpy(mapped, b, size_b);
    vkUnmapMemory(device, mem_b);

    // Descriptor set
    VkDescriptorSet ds;
    VkDescriptorSetAllocateInfo dsai = {};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    vkAllocateDescriptorSets(device, &dsai, &ds);

    VkDescriptorBufferInfo dbi[3] = {};
    dbi[0].buffer = buf_a; dbi[0].offset = 0; dbi[0].range = size_a;
    dbi[1].buffer = buf_b; dbi[1].offset = 0; dbi[1].range = size_b;
    dbi[2].buffer = buf_c; dbi[2].offset = 0; dbi[2].range = size_c;

    VkWriteDescriptorSet writes[3] = {};
    for (int i = 0; i < 3; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = ds;
      writes[i].dstBinding = static_cast<std::uint32_t>(i);
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

    // Command buffer
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;

    VkCommandBuffer cb;
    vkAllocateCommandBuffers(device, &cbai, &cb);

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &cbbi);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &ds, 0, nullptr);

    std::int32_t params[3] = { M, K, N };
    vkCmdPushConstants(cb, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), params);

    const std::uint32_t gx = (static_cast<std::uint32_t>(N) + 15) / 16;
    const std::uint32_t gy = (static_cast<std::uint32_t>(M) + 15) / 16;
    vkCmdDispatch(cb, gx, gy, 1);

    vkEndCommandBuffer(cb);

    // Submit
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(queue, 1, &si, nullptr);
    vkQueueWaitIdle(queue);

    // Read back C
    vkMapMemory(device, mem_c, 0, size_c, 0, &mapped);
    std::memcpy(c, mapped, size_c);
    vkUnmapMemory(device, mem_c);

    // Cleanup per-call resources
    vkFreeCommandBuffers(device, cmd_pool, 1, &cb);
    vkDestroyBuffer(device, buf_a, nullptr); vkFreeMemory(device, mem_a, nullptr);
    vkDestroyBuffer(device, buf_b, nullptr); vkFreeMemory(device, mem_b, nullptr);
    vkDestroyBuffer(device, buf_c, nullptr); vkFreeMemory(device, mem_c, nullptr);
  }

  ~Impl() {
    if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
    if (pipe_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipe_layout, nullptr);
    if (dsl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    if (desc_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, desc_pool, nullptr);
    if (cmd_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, cmd_pool, nullptr);
    if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
  }
};

VulkanBackend::VulkanBackend() {
  auto impl = new Impl();
  if (impl->initialise()) {
    instance_ = impl->instance;
    device_ = impl->device;
    impl_ = impl;
  } else {
    delete impl;
  }
}

VulkanBackend::~VulkanBackend() {
  if (impl_) {
    Impl* p = static_cast<Impl*>(impl_);
    delete p;
  }
}

void VulkanBackend::matmul2d_fwd(int m, int k, int n, const float* a, const float* b, float* c) {
  if (impl_) {
    static_cast<Impl*>(impl_)->run_matmul_fwd(m, k, n, a, b, c);
  } else {
    // Fallback to CPU
    CpuBackend cpu;
    cpu.matmul2d_fwd(m, k, n, a, b, c);
  }
}

void VulkanBackend::matmul2d_bwd(int m, int k, int n,
                                  const float* a_mk, const float* b_kn,
                                  const float* d_out_mn,
                                  float* d_a_mk, float* d_b_kn) {
  // POC: backward not yet on GPU — delegate to CPU reference
  CpuBackend cpu;
  cpu.matmul2d_bwd(m, k, n, a_mk, b_kn, d_out_mn, d_a_mk, d_b_kn);
}

#else  // !BUILD_VULKAN

// Stub: Vulkan not enabled at build time
VulkanBackend::VulkanBackend() {}
VulkanBackend::~VulkanBackend() {}

void VulkanBackend::matmul2d_fwd(int m, int k, int n, const float* a, const float* b, float* c) {
  CpuBackend cpu;
  cpu.matmul2d_fwd(m, k, n, a, b, c);
}

void VulkanBackend::matmul2d_bwd(int m, int k, int n,
                                  const float* a_mk, const float* b_kn,
                                  const float* d_out_mn,
                                  float* d_a_mk, float* d_b_kn) {
  CpuBackend cpu;
  cpu.matmul2d_bwd(m, k, n, a_mk, b_kn, d_out_mn, d_a_mk, d_b_kn);
}

#endif // BUILD_VULKAN

} // namespace backend
