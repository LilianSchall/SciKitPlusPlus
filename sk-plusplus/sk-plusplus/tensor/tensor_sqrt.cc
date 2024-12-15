#include "tensor.hh"
#include <cmath>

namespace sk::tensor
{
sk::Tensor sqrt(const sk::Tensor &t)
{
    sk::Tensor output{t};

    return output.map([](float x) {return std::sqrt(x);});
}
} // namespace sk::tensor
