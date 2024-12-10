#pragma once

#include "sk-plusplus/nn/module.hh"
#include "sk-plusplus/tensor/tensor.hh"

#include "exp.hh"

namespace sk::nn
{
class Sigmoid : public sk::nn::Module
{
  public:
    explicit Sigmoid() = default;

    sk::Tensor forward(sk::Tensor &input) override;
    // sk::Tensor backward(sk::Tensor &input) override;

  private:
    Exp _e;
};
} // namespace sk::nn
