#pragma once

#include "serializer.hh"
#include "sk-plusplus/tensor/tensor.hh"

namespace sk::serializer
{

class TensorSerializer : Serializer<sk::Tensor>
{
  public:
    void serialize(const sk::Tensor &t, const std::string &filepath) override;
    sk::Tensor deserialize(const std::string &filepath) override;
};

} // namespace sk::serializer
