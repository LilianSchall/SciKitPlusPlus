#include "scaler.hh"

namespace sk::scaler
{
sk::Tensor StandardScaler::transform(sk::Tensor &input) 
{ 
    sk::Tensor output = input - sk::tensor::mean(input, 1);

    output /= sk::tensor::std(input, 1);

    return output; 
}
} // namespace sk::scaler
