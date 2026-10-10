# 作业六 —— 卡尔曼滤波（装甲板位姿滤波）
#
# 原来的 `cmake_minimum_required(VERSION 4.3)` 是不存在的版本号，
# 会让任何人 clone 下来在 configure 阶段就失败，这里改成实际存在且够用的 3.16。
cmake_minimum_required(VERSION 3.16)

project(l6 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(l6
    main.cpp
    src/kalman.cpp
    src/data.cpp
)

target_include_directories(l6 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(l6 PRIVATE /W4 /utf-8)
else()
    target_compile_options(l6 PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 可执行文件留在构建目录里（默认行为），不要写回源码目录：
# 源码目录里的 ./l6 由 run.sh 生成，两条构建路径互不干扰。
