cmake_minimum_required(VERSION 3.22)

target_sources(test PRIVATE
    tensor_test.cc
    tensor_add_test.cc
    tensor_mul_test.cc
    mse_loss_test.cc
)
