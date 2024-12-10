#include "exp.hh"
#include "sk-plusplus/tensor/tensor.hh"
#include <math.h>

namespace sk::nn
{

sk::Tensor Exp::forward(sk::Tensor &input)
{
    sk::Tensor output{input};
    return output.map([](float x){return expf(x);});
}
} // namespace sk::nn
