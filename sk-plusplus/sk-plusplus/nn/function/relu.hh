#pragma once

#include "sk-plusplus/nn/module.hh"
#include "sk-plusplus/tensor/tensor.hh"

#include "exp.hh"

namespace sk::nn
{
class Relu : public sk::nn::Module
{
  public:
    explicit Relu() = default;

    sk::Tensor forward(sk::Tensor &input) override;
    // sk::Tensor backward(sk::Tensor &input) override;

  private:
    Exp _e;
};
} // namespace sk::nn
