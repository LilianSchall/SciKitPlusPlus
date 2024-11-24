cmake_minimum_required(VERSION 3.22)

target_sources(test PRIVATE
    tensor_test.cc
    tensor_add.cc
    tensor_mul.cc
)
