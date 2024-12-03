#pragma once

#include "sk-plusplus/tensor/tensor.hh"

namespace sk::nn::loss
{
class Loss
{
  public:
    virtual sk::Tensor
    forward(sk::Tensor &target, sk::Tensor &prediction) = 0;
    // sk::Tensor backward(sk::Tensor &input) override;
};
} // namespace sk::nn::loss
