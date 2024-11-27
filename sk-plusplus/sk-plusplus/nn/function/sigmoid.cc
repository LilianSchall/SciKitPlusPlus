#include "sigmoid.hh"
#include "sk-plusplus/tensor/tensor.hh"

namespace sk::nn
{

sk::Tensor Sigmoid::forward(sk::Tensor &input)
{
    sk::Tensor e = this->_e.forward(input);

    e += 1;

    e.map([](float x) { return 1 / x; });

    return e;
}
} // namespace sk::nn
