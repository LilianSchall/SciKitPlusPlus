cmake_minimum_required(VERSION 3.22)

target_sources(test PRIVATE
    mse_loss_test.cc
    tensor_add_test.cc
    tensor_mul_test.cc
    tensor_serializer_test.cc
    tensor_sub_test.cc
    tensor_test.cc
)
configure_file(examples/x.dat ${CMAKE_CURRENT_BINARY_DIR}/../examples/x.dat)
configure_file(examples/y.dat ${CMAKE_CURRENT_BINARY_DIR}/../examples/y.dat)
configure_file(examples/weights.dat ${CMAKE_CURRENT_BINARY_DIR}/../examples/weights.dat)
configure_file(examples/biases.dat ${CMAKE_CURRENT_BINARY_DIR}/../examples/biases.dat)
