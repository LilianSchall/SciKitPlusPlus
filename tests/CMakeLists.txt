cmake_minimum_required(VERSION 3.22)

target_sources(test PRIVATE
    logistic_regression_test.cc
    min_max_scaler_test.cc
    mnist_test.cc
    mse_loss_test.cc
    standard_scaler_test.cc
    tensor_add_test.cc
    tensor_div_test.cc
    tensor_max_test.cc
    tensor_mean_test.cc
    tensor_min_test.cc
    tensor_mul_test.cc
    tensor_serializer_test.cc
    tensor_split_test.cc
    tensor_sub_test.cc
    tensor_sum_test.cc
    tensor_test.cc
    tensor_var_test.cc
    kmeans_test.cc
)
# configure_file(examples/x.dat ${CMAKE_CURRENT_BINARY_DIR}/../examples/x.dat)
# configure_file(examples/y.dat ${CMAKE_CURRENT_BINARY_DIR}/../examples/y.dat)
# configure_file(examples/weights.dat ${CMAKE_CURRENT_BINARY_DIR}/../examples/weights.dat)
# configure_file(examples/biases.dat ${CMAKE_CURRENT_BINARY_DIR}/../examples/biases.dat)
