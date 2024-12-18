#pragma once

#include "sk-plusplus/nn/module.hh"
#include "sk-plusplus/tensor/tensor.hh"
#include <cstddef>

namespace sk::nn
{
class Linear : public sk::nn::Module
{
  public:
    explicit Linear(sk::Tensor weights, sk::Tensor bias);
    explicit Linear(size_t input_size, size_t output_size);

    sk::Tensor forward(sk::Tensor &input) override;
    // sk::Tensor backward(sk::Tensor &input) override;

  private:
    sk::Tensor _weights;
    sk::Tensor _bias;
};
} // namespace sk::nn
