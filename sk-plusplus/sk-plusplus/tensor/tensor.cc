#include "tensor.hh"
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

    std::vector<float> data{dimension, value, std::allocator<float>{}};

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

float& sk::Tensor::operator[](const std::vector<size_t>& indices)
{
    size_t index = 0;

    for (size_t i = 0; i < this->shape.size() - 1; i++)
        index = this->shape[i + 1] * indices[i];

    return this->_data[index];
}

std::vector<size_t> sk::Tensor::create_iter(void)
{
    std::vector<size_t> iter{this->shape.size(), 0, std::allocator<size_t>{}};

    return iter;
}

