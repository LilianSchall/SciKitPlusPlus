#pragma once

#include "sk-plusplus/tensor/tensor.hh"

namespace sk::cluster
{
class KMeans
{
  public:
    explicit KMeans(sk::Tensor &centroids);
    sk::Tensor predict(sk::Tensor &input);

  private:
    sk::Tensor centroids_;
};

} // namespace sk::cluster
