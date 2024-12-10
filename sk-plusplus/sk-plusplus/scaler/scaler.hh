#pragma once

#include "sk-plusplus/tensor/tensor.hh"

namespace sk::scaler
{

class Scaler
{
  public:
    virtual sk::Tensor transform(sk::Tensor &input) = 0;
    virtual sk::Tensor inverse_transform(sk::Tensor &input) = 0;
};

class MinMaxScaler : Scaler
{
  public:
    MinMaxScaler(sk::Tensor min, sk::Tensor max) = delete;
    MinMaxScaler(sk::Tensor &input);
    sk::Tensor transform(sk::Tensor &input) override;
    sk::Tensor inverse_transform(sk::Tensor &input) override;

  private:
    sk::Tensor _min;
    sk::Tensor _max;
};

class StandardScaler : Scaler
{
  public:
    StandardScaler(sk::Tensor mean, sk::Tensor std) = delete;
    StandardScaler(sk::Tensor &input);
    sk::Tensor transform(sk::Tensor &input) override;
    sk::Tensor inverse_transform(sk::Tensor &input) override;

  private:
    sk::Tensor _mean;
    sk::Tensor _std;
};
} // namespace sk::scaler
