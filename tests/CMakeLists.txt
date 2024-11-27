cmake_minimum_required(VERSION 3.22)

target_sources(test PRIVATE
    mse_loss_test.cc
    tensor_add_test.cc
    tensor_mul_test.cc
    tensor_serializer_test.cc
    tensor_sub_test.cc
    tensor_test.cc
)
