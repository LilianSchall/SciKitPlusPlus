#pragma once

#include "sk-plusplus/tensor/tensor.hh"
namespace sk::nn
{
class Module
{
  public:
    virtual sk::Tensor forward(sk::Tensor &input) = 0;
    // virtual sk::Tensor backward(sk::Tensor &input) = 0;
};
} // namespace sk::nn
