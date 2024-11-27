#include "exp.hh"
#include <math.h>

namespace sk::nn
{

sk::Tensor Exp::forward(sk::Tensor &input)
{
    return input.map([](float x){return expf(x);});
}
} // namespace sk::nn
