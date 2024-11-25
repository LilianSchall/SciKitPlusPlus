#pragma once

#include "sk-plusplus/nn/module.hh"
#include "sk-plusplus/tensor/tensor.hh"

namespace sk::nn::loss
{
class MSELoss // : sk::nn::Module
{
  public:
    explicit MSELoss() = default;

    sk::Tensor forward(sk::Tensor &target, sk::Tensor &prediction);
    // sk::Tensor backward(sk::Tensor &input) override;
};
} // namespace sk::nn::loss
