cmake_minimum_required(VERSION 3.9)
project(armor_pnp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui calib3d dnn)

add_executable(armor_pnp
        src/main.cpp
        src/armor_detector.cpp
        src/tracker.cpp
        src/pnp_solver.cpp
        src/visualizer.cpp
)

target_include_directories(armor_pnp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(armor_pnp ${OpenCV_LIBS})
