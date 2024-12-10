cmake_minimum_required(VERSION 3.22)

target_sources(${PROJECT_NAME} PRIVATE
    linear.cc
    logistic_regression.cc
    sequential.cc
)

add_subdirectory(loss/)
add_subdirectory(function/)
