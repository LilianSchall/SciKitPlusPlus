#pragma once

#include "sk-plusplus/nn/module.hh"
#include "sk-plusplus/tensor/tensor.hh"

namespace sk::nn
{
class Flatten : public sk::nn::Module
{
  public:
    explicit Flatten() = default;

    sk::Tensor forward(sk::Tensor &input) override;
    // sk::Tensor backward(sk::Tensor &input) override;
};
} // namespace sk::nn
