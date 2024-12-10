#include "scaler.hh"

namespace sk::scaler
{
sk::Tensor MinMaxScaler::transform(sk::Tensor &input)
{
    sk::Tensor min = sk::tensor::min(input, 1);
    sk::Tensor max = sk::tensor::max(input, 1);
    sk::Tensor output = input - min;

    output /= (max - min);

    return output;
}
} // namespace sk::scaler
