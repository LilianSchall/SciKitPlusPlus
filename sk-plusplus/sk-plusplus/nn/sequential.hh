#pragma once

#include "sk-plusplus/nn/module.hh"
#include "sk-plusplus/tensor/tensor.hh"
#include <memory>
#include <vector>

namespace sk::nn
{
class Sequential : public sk::nn::Module
{
  public:
    explicit Sequential();
    explicit Sequential(std::vector<std::shared_ptr<sk::nn::Module>> &layers);

    void add_layer(std::shared_ptr<sk::nn::Module> layer);

    sk::Tensor forward(sk::Tensor &input) override;
    // sk::Tensor backward(sk::Tensor &input) override;

  private:
    std::vector<std::shared_ptr<sk::nn::Module>> _layers;
};
} // namespace sk::nn
