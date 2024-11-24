#include "linear.hh"

namespace sk::nn
{

Linear::Linear(size_t input_size, size_t output_size):
    _weights(sk::Tensor::zeroes({input_size, output_size})),
    _bias(sk::Tensor::zeroes({input_size, output_size}))
{}

sk::Tensor Linear::forward(sk::Tensor& input)
{
    sk::Tensor res = this->_weights * input;

    res += this->_bias;

    return res;
}
}
