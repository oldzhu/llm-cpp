cmake_minimum_required(VERSION 3.20)
project(build_llm_using_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
)

target_include_directories(llm_core PUBLIC src)

add_library(llm_backend INTERFACE)
target_include_directories(llm_backend INTERFACE src)

add_library(llm_variant_mha
  src/variants/mha/mha_attention.cpp
)
target_include_directories(llm_variant_mha PUBLIC src)
target_link_libraries(llm_variant_mha PUBLIC llm_core)

add_executable(train_gpt
  src/main.cpp
)

target_link_libraries(train_gpt PRIVATE llm_core)

enable_testing()

add_executable(test_build_llm
  tests/test_main.cpp
)

target_link_libraries(test_build_llm PRIVATE llm_core llm_variant_mha)

add_test(NAME build_llm_tests COMMAND test_build_llm)

if (MSVC)
  target_compile_options(llm_core PRIVATE /W4 /permissive-)
  target_compile_options(llm_variant_mha PRIVATE /W4 /permissive-)
  target_compile_options(train_gpt PRIVATE /W4 /permissive-)
  target_compile_options(test_build_llm PRIVATE /W4 /permissive-)
else()
  target_compile_options(llm_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(llm_variant_mha PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(train_gpt PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(test_build_llm PRIVATE -Wall -Wextra -Wpedantic)
endif()
