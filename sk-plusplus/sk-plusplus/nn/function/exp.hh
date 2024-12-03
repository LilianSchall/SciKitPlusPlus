#pragma once

#include "sk-plusplus/nn/module.hh"
#include "sk-plusplus/tensor/tensor.hh"
#include <cstddef>

namespace sk::nn
{
class Exp : sk::nn::Module
{
  public:
    explicit Exp() = default;

    sk::Tensor forward(sk::Tensor &input) override;
    // sk::Tensor backward(sk::Tensor &input) override;
};
} // namespace sk::nn
