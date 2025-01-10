#pragma once

#include "sk-plusplus/nn/module.hh"
#include "sk-plusplus/tensor/tensor.hh"

#include "exp.hh"

namespace sk::nn
{
class Softmax : public sk::nn::Module
{
  public:
    explicit Softmax() = default;

    sk::Tensor forward(sk::Tensor &input) override;
    // sk::Tensor backward(sk::Tensor &input) override;

  private:
    Exp _e;
};
} // namespace sk::nn
