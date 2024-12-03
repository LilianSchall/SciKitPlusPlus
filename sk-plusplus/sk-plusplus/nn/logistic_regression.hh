#pragma once

#include "sk-plusplus/nn/linear.hh"
#include "sk-plusplus/nn/function/sigmoid.hh"
#include "sk-plusplus/nn/module.hh"
#include "sk-plusplus/tensor/tensor.hh"
#include <cstddef>

namespace sk::nn
{
class LogisticRegression : sk::nn::Module
{
  public:
    explicit LogisticRegression(sk::Tensor weights, sk::Tensor bias);
    explicit LogisticRegression(size_t input_size, size_t output_size);

    sk::Tensor forward(sk::Tensor &input) override;
    // sk::Tensor backward(sk::Tensor &input) override;

  private:
    Linear _fc;
    Sigmoid _sigmoid;
};
} // namespace sk::nn
