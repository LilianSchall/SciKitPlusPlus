#include "scaler.hh"

namespace sk::scaler
{

MinMaxScaler::MinMaxScaler(sk::Tensor &input) :
    _min(sk::tensor::min(input, 1)), _max(sk::tensor::max(input, 1))

{
}

sk::Tensor MinMaxScaler::transform(sk::Tensor &input)
{
    sk::Tensor output = input - this->_min;

    output /= (this->_max - this->_min);

    return output;
}

sk::Tensor MinMaxScaler::inverse_transform(sk::Tensor &input)
{
    sk::Tensor scaled = (this->_max - this->_min);
    scaled += this->_min;
    sk::Tensor output = input.hadamard_dot(scaled);

    return output;
}
} // namespace sk::scaler
