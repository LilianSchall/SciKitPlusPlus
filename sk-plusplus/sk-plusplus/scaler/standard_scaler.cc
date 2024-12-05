#include "scaler.hh"

namespace sk::scaler
{
sk::Tensor StandardScaler::transform(sk::Tensor &input) 
{ 
    sk::Tensor output = input - sk::tensor::mean(input);

    output /= sk::tensor::std(input);

    return output; 
}
} // namespace sk::scaler
