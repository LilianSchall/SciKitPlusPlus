#include "scaler.hh"
#include "sk-plusplus/tensor/tensor.hh"

namespace sk::scaler
{

StandardScaler::StandardScaler(sk::Tensor &input) :
    _mean(sk::tensor::mean(input, 1)), _std(sk::tensor::std(input, 1))
{
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
