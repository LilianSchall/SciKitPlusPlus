#include "flatten.hh"
#include <numeric>

namespace sk::nn
{

sk::Tensor Flatten::forward(sk::Tensor &input)
{
    size_t size = std::accumulate(input.shape.cbegin(), input.shape.cend(), 1, std::multiplies<int>());
    return input.reshape({size});
}
} // namespace sk::nn
