cmake_minimum_required(VERSION 3.22)

add_subdirectory(engine/)
add_subdirectory(tools/)

configure_file(assets/Sans.ttf ${CMAKE_CURRENT_BINARY_DIR}/../Sans.ttf COPYONLY)

target_sources(mnist PRIVATE
    mnist.cc
)
