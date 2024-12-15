cmake_minimum_required(VERSION 3.22)

target_sources(${PROJECT_NAME} PRIVATE
    tensor.cc
    tensor_add.cc
    tensor_argmax.cc
    tensor_argmin.cc
    tensor_div.cc
    tensor_max.cc
    tensor_mean.cc
    tensor_min.cc
    tensor_mul.cc
    tensor_print.cc
    tensor_split.cc
    tensor_sqrt.cc
    tensor_std.cc
    tensor_sub.cc
    tensor_sum.cc
    tensor_var.cc
)
