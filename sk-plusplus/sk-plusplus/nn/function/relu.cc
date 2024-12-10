#include "relu.hh"

namespace sk::nn
{

sk::Tensor Relu::forward(sk::Tensor &input)
{
    sk::Tensor output{input};
    return output.map([](float x) {return x > 0 ? x : 0;});
}
} // namespace sk::nn
