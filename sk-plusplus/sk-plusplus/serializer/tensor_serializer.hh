#pragma once

#include "serializer.hh"

namespace sk::serializer
{

class TensorSerializer : Serializer
{
  public:
    void serialize(const sk::Tensor &t, const std::string &filepath) override;
    sk::Tensor deserialize(const std::string &filepath) override;
};

} // namespace sk::serializer
