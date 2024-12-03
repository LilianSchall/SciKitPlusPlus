#include "logistic_regression.hh"

namespace sk::nn
{

LogisticRegression::LogisticRegression(sk::Tensor weights, sk::Tensor bias) :
    _fc(weights, bias), _sigmoid()
{
}

LogisticRegression::LogisticRegression(size_t input_size, size_t output_size) :
    _fc(input_size, output_size), _sigmoid()
{
}

sk::Tensor LogisticRegression::forward(sk::Tensor &input)
{
    sk::Tensor r = this->_fc.forward(input);

    r = this->_sigmoid.forward(r);
    return sk::tensor::argmax(r, 1);
}
} // namespace sk::nn
