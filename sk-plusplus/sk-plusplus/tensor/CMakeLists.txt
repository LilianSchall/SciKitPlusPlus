cmake_minimum_required(VERSION 3.22)

target_sources(${PROJECT_NAME} PRIVATE
    tensor.cc
    tensor_argmax.cc
    tensor_print.cc
    tensor_sum.cc
    tensor_add.cc
    tensor_sub.cc
    tensor_mul.cc
)
