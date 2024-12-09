#include "tensor.hh"

namespace sk::tensor
{
sk::Tensor var(sk::Tensor &t, int axis, int ddof)
{
    sk::Tensor tensor_mean = sk::tensor::mean(t, axis);

    sk::Tensor items = (axis == 1 ? sk::tensor::transpose(t) : t) - tensor_mean;

    items = items.hadamard_dot(items);

    sk::Tensor out = sk::tensor::sum(items,axis);

    int n = (axis == -1 ? t.as_array().size() : t.shape[axis]);
    out = out.map([ddof, n](float x){ return x / (n - ddof);});

    return out;
}
} // namespace sk::tensor
