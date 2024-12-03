#pragma once

#include "sk-plusplus/nn/loss/loss.hh"
#include "sk-plusplus/tensor/tensor.hh"

namespace sk::nn::loss
{
class MSELoss : sk::nn::loss::Loss
{
  public:
    explicit MSELoss() = default;

    sk::Tensor forward(sk::Tensor &target, sk::Tensor &prediction) override;
    // sk::Tensor backward(sk::Tensor &input) override;
};
} // namespace sk::nn::loss
