cmake_minimum_required(VERSION 3.20)
project(build_llm_using_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(train_gpt
  src/main.cpp
  src/tensor.cpp
  src/ops.cpp
  src/model.cpp
  src/optim.cpp
  src/checkpoint.cpp
  src/data.cpp
  src/util.cpp
)

target_include_directories(train_gpt PRIVATE src)

if (MSVC)
  target_compile_options(train_gpt PRIVATE /W4 /permissive-)
else()
  target_compile_options(train_gpt PRIVATE -Wall -Wextra -Wpedantic)
endif()
