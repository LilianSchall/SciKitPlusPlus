cmake_minimum_required(VERSION 3.22)

target_sources(mnist PRIVATE
    sdl_tools.cc
    build_model.cc
)
