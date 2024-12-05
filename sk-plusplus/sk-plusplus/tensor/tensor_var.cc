#include "tensor.hh"
#include <numeric>
#include <vector>

namespace sk::tensor
{
sk::Tensor var(const sk::Tensor &t, int axis)
{
    sk::Tensor tensor_mean = sk::tensor::mean(t, axis);

    sk::Tensor sub = sk::tensor::sum(
        (axis == 0 ? sk::tensor::transpose(t) : t) - tensor_mean,
        axis);

    sk::Tensor output = sub.hadamard_dot(sub);
}
} // namespace sk::tensor
