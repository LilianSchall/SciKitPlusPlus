#include "softmax.hh"
#include "sk-plusplus/tensor/tensor.hh"

namespace sk::nn
{

sk::Tensor Softmax::forward(sk::Tensor &input)
{
    sk::Tensor e = this->_e.forward(input);
    sk::Tensor sum = sk::tensor::sum(e, input.shape.size() - 1);

    sk::Tensor output = e / sum;

    return output;
}
} // namespace sk::nn
