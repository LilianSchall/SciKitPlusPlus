#include "tensor.hh"
#include <cmath>

namespace sk::tensor
{
sk::Tensor std(sk::Tensor &t, int axis, int ddof)
{
    sk::Tensor tensor_var = sk::tensor::var(t, axis, ddof);
    
    tensor_var = tensor_var.map([](float x) {return std::sqrt(x);});

    return tensor_var;
}
} // namespace sk::tensor
