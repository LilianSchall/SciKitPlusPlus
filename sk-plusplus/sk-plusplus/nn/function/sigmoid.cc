#include "sigmoid.hh"
#include "sk-plusplus/tensor/tensor.hh"

namespace sk::nn
{

sk::Tensor Sigmoid::forward(sk::Tensor &input)
{
    sk::Tensor neg = -input;

    sk::Tensor e = this->_e.forward(neg);

    e += 1;
    e = 1 / e;

    return e;
}
} // namespace sk::nn
