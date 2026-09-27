cmake_minimum_required(VERSION 3.22)
project(l2)

set(CMAKE_CXX_STANDARD 17)

find_package(zenohc)
find_package(zenohcxx)
find_package(OpenCV)

add_executable(publisher publisher.cpp)
target_link_libraries(publisher PRIVATE zenohcxx::zenohc)
target_link_libraries(publisher PRIVATE ${OpenCV_LIBRARIES})
target_link_directories(publisher PRIVATE ${OpenCV_DIRECTORIES})

add_executable(subscriber subscriber.cpp)
target_link_libraries(subscriber PRIVATE zenohcxx::zenohc)
target_link_libraries(subscriber PRIVATE ${OpenCV_LIBRARIES})
target_link_directories(subscriber PRIVATE ${OpenCV_DIRECTORIES})
