#include "scaler.hh"
#include "sk-plusplus/tensor/tensor.hh"

namespace sk::scaler
{

StandardScaler::StandardScaler(sk::Tensor &input) :
    _mean(sk::tensor::mean(input, 0)), _std(sk::tensor::std(input, 0))
{
    assert(
        input.shape.size() == 2 &&
        "Expected StandardScaler input to be a 2D array");
}

sk::Tensor StandardScaler::transform(sk::Tensor &input)
{
    sk::Tensor output = input - this->_mean;

    output /= this->_std;

    return output;
}

sk::Tensor StandardScaler::inverse_transform(sk::Tensor &input)
{
    sk::Tensor output = input.hadamard_dot(this->_std);

    output += this->_mean;

    return output;
}
} // namespace sk::scaler
