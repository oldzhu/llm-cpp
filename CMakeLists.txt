cmake_minimum_required(VERSION 3.20)
project(build_llm_using_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_VULKAN "Enable Vulkan compute backend (requires Vulkan SDK)" OFF)

add_library(llm_core
  src/tensor.cpp
  src/ops.cpp
  src/model.cpp
  src/optim.cpp
  src/checkpoint.cpp
  src/data.cpp
  src/util.cpp

  src/backend/cpu_backend.cpp
  src/backend/registry.cpp
  src/backend/blocked_simd_backend.cpp

  src/tokenizer/byte_tokenizer.cpp
  src/tokenizer/bpe_tokenizer.cpp
)

if (BUILD_VULKAN)
  find_package(Vulkan REQUIRED)
  target_sources(llm_core PRIVATE src/backend/vulkan/vulkan_backend.cpp)
  target_include_directories(llm_core PUBLIC ${Vulkan_INCLUDE_DIRS})
  target_link_libraries(llm_core PUBLIC ${Vulkan_LIBRARY})
  target_compile_definitions(llm_core PUBLIC BUILD_VULKAN)

  # Compile GLSL shader to SPIR-V at build time
  find_program(GLSLC glslc PATHS "$ENV{VULKAN_SDK}/Bin" "${Vulkan_GLSLC_EXECUTABLE}" ENV VULKAN_SDK)
  if (GLSLC)
    add_custom_command(
      OUTPUT "${CMAKE_BINARY_DIR}/shaders/matmul_fwd.spv"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
      COMMAND ${GLSLC} "${CMAKE_SOURCE_DIR}/shaders/matmul_fwd.comp" -o "${CMAKE_BINARY_DIR}/shaders/matmul_fwd.spv"
      DEPENDS "${CMAKE_SOURCE_DIR}/shaders/matmul_fwd.comp"
      COMMENT "Compiling Vulkan shader: matmul_fwd.comp → matmul_fwd.spv"
    )
    add_custom_target(vulkan_shaders ALL DEPENDS "${CMAKE_BINARY_DIR}/shaders/matmul_fwd.spv")
  else()
    message(WARNING "Vulkan SDK found but glslc not found. Shaders must be compiled manually:\n"
                    "  glslc shaders/matmul_fwd.comp -o shaders/matmul_fwd.spv")
  endif()
endif()

target_include_directories(llm_core PUBLIC src)

add_library(llm_backend INTERFACE)
target_include_directories(llm_backend INTERFACE src)

add_library(llm_variant_mha
  src/variants/mha/mha_attention.cpp
)
target_include_directories(llm_variant_mha PUBLIC src)
target_link_libraries(llm_variant_mha PUBLIC llm_core)

add_library(llm_variant_kvcache
  src/variants/kvcache/kvcache_attention.cpp
)
target_include_directories(llm_variant_kvcache PUBLIC src)
target_link_libraries(llm_variant_kvcache PUBLIC llm_core)

add_library(llm_variant_rope
  src/variants/rope/rope_attention.cpp
)
target_include_directories(llm_variant_rope PUBLIC src)
target_link_libraries(llm_variant_rope PUBLIC llm_core)

add_library(llm_variant_gqa
  src/variants/gqa/gqa_attention.cpp
)
target_include_directories(llm_variant_gqa PUBLIC src)
target_link_libraries(llm_variant_gqa PUBLIC llm_core)

add_executable(train_gpt
  src/main.cpp
)

add_library(llm_variant_moe
  src/variants/moe/moe_mlp.cpp
)
target_include_directories(llm_variant_moe PUBLIC src)
target_link_libraries(llm_variant_moe PUBLIC llm_core)

add_library(llm_variant_mla
  src/variants/mla/mla_attention.cpp
)
target_include_directories(llm_variant_mla PUBLIC src)
target_link_libraries(llm_variant_mla PUBLIC llm_core)

target_link_libraries(train_gpt PRIVATE llm_core llm_variant_kvcache llm_variant_moe llm_variant_mha llm_variant_gqa llm_variant_rope llm_variant_mla)

enable_testing()

add_executable(test_build_llm
  tests/test_main.cpp
)

target_link_libraries(test_build_llm PRIVATE llm_core llm_variant_mha llm_variant_kvcache llm_variant_rope llm_variant_gqa llm_variant_moe llm_variant_mla)

add_test(NAME build_llm_tests COMMAND test_build_llm WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

if (MSVC)
  target_compile_options(llm_core PRIVATE /W4 /permissive-)
  target_compile_options(llm_variant_mha PRIVATE /W4 /permissive-)
  target_compile_options(llm_variant_kvcache PRIVATE /W4 /permissive-)
  target_compile_options(llm_variant_rope PRIVATE /W4 /permissive-)
  target_compile_options(llm_variant_gqa PRIVATE /W4 /permissive-)
  target_compile_options(llm_variant_moe PRIVATE /W4 /permissive-)
  target_compile_options(llm_variant_mla PRIVATE /W4 /permissive-)
  target_compile_options(train_gpt PRIVATE /W4 /permissive-)
  target_compile_options(test_build_llm PRIVATE /W4 /permissive-)
else()
  target_compile_options(llm_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(llm_variant_mha PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(llm_variant_kvcache PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(llm_variant_rope PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(llm_variant_gqa PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(llm_variant_moe PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(llm_variant_mla PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(train_gpt PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(test_build_llm PRIVATE -Wall -Wextra -Wpedantic)
endif()
