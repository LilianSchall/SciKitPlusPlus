#include "tensor.hh"
#include <memory>
#include <numeric>
#include <vector>

namespace sk
{
Tensor::Tensor(std::vector<float> data, std::vector<size_t> shape) :
    shape(shape), _data(data)
{
}

Tensor operator==(const Tensor &lhs, const Tensor &rhs)
{
    const std::vector<float> &l = lhs.as_array();
    const std::vector<float> &r = rhs.as_array();

    const size_t n = l.size();

    std::vector<float> data{ n, 0, std::allocator<float>{} };

    for (size_t i = 0; i < n; i++)
        data[i] = l[i] == r[i];

    return sk::Tensor{ data, lhs.shape };
}

const std::vector<float> &Tensor::as_array() const { return this->_data; }

Tensor &Tensor::reshape(std::vector<size_t> shape)
{
    const size_t dimension = std::accumulate(
        shape.cbegin(),
        shape.cend(),
        1,
        std::multiplies<size_t>{});

    assert(
        dimension == this->_data.size() &&
        "Given shape does not fully match data size");

    this->shape = shape;

    return *this;
}

Tensor &Tensor::map(std::function<float(float)> func)
{
    for (size_t i = 0; i < this->_data.size(); i++)
        this->_data[i] = func(this->_data[i]);
    return *this;
}

Tensor &Tensor::transpose(void)
{
    if (this->shape.size() > 2)
        return *this;

    std::vector<float> data;

    if (this->shape.size() == 2)
    {
        for (size_t j = 0; j < this->shape[1]; j++)
            for (size_t i = 0; i < this->shape[0]; i++)
                data.push_back(this->_data[i * this->shape[1] + j]);
    }

    size_t dim = *(this->shape.end() - 1);
    this->shape.pop_back();

    this->shape.emplace(this->shape.begin(), dim);

    if (this->shape.size() == 1)
        this->shape.emplace(this->shape.begin(), 1);

    this->_data = data;

    return *this;
}

} // namespace sk

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

namespace sk::tensor
{
sk::Tensor ones(std::vector<size_t> shape) { return tensor_fill(shape, 1.0); }

sk::Tensor zeroes(std::vector<size_t> shape) { return tensor_fill(shape, 0.0); }

sk::Tensor arange(int max, int min, int step)
{
    std::vector<float> data;

    for (int i = min; i < max; i += step)
        data.push_back(i);

    return sk::Tensor{ data, { data.size() } };
}
} // namespace sk::tensor
