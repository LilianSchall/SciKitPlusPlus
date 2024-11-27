#pragma once

#include <string>

#include "sk-plusplus/tensor/tensor.hh"

namespace sk::serializer
{

template <typename Data>
class Serializer
{
  public:
    virtual void
    serialize(const Data &t, const std::string &filepath) = 0;
    virtual Data deserialize(const std::string &filepath) = 0;
};

} // namespace sk::serializer
