#pragma once

#include "sk-plusplus/tensor/tensor.hh"

namespace sk::scaler
{

class Scaler
{
  public:
    virtual sk::Tensor transform(sk::Tensor &input) = 0;
};

class MinMaxScaler : Scaler
{
  public:
    sk::Tensor transform(sk::Tensor &input) override;
};

class StandardScaler : Scaler
{
  public:
    sk::Tensor transform(sk::Tensor &input) override;
};
} // namespace sk::scaler
