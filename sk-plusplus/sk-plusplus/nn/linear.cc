#include "linear.hh"
#include <vector>

namespace sk::nn
{

Linear::Linear(sk::Tensor weights, sk::Tensor bias) :
    _weights(weights), _bias(bias)
{
}

Linear::Linear(size_t input_size, size_t output_size) :
    _weights(sk::tensor::zeroes({ input_size, output_size })),
    _bias(sk::tensor::zeroes({ input_size, output_size }))
{
}

sk::Tensor Linear::forward(sk::Tensor &input)
{
    sk::Tensor res = input * this->_weights;

    res += this->_bias;

    // return sk::tensor::argmax(res, axis=1);
    return res;
}
} // namespace sk::nn
