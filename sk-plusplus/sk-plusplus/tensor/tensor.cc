#include "tensor.hh"
#include <cassert>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

sk::Tensor::Tensor(std::vector<float> data, std::vector<size_t> shape) :
    shape(shape), _data(data)
{
}

sk::Tensor tensor_fill(std::vector<size_t> shape, float value)
{
    const size_t dimension = std::accumulate(
        shape.cbegin(),
        shape.cend(),
        1,
        std::multiplies<size_t>{});

    std::vector<float> data{ dimension, value, std::allocator<float>{} };

    return sk::Tensor(data, shape);
}

sk::Tensor sk::Tensor::ones(std::vector<size_t> shape)
{
    return tensor_fill(shape, 1.0);
}

sk::Tensor sk::Tensor::zeroes(std::vector<size_t> shape)
{
    return tensor_fill(shape, 0.0);
}
