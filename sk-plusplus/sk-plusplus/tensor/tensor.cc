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
    sk::tensor::transpose(*this, *this);
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

sk::Tensor transpose(const sk::Tensor &a)
{
    sk::Tensor result = sk::tensor::zeroes(a.shape);

    transpose(a, result);

    return result;
}

void transpose(const sk::Tensor &a, sk::Tensor &result)
{
    if (a.shape.size() > 2)
        return;

    std::vector<float> data;

    std::vector<float> a_data = a.as_array();

    if (a.shape.size() == 2)
    {
        for (size_t j = 0; j < a.shape[1]; j++)
            for (size_t i = 0; i < a.shape[0]; i++)
                data.push_back(a_data[i * a.shape[1] + j]);
    }

    size_t dim = *(a.shape.end() - 1);
    result.shape = a.shape;
    result.shape.pop_back();


    result.shape.emplace(result.shape.begin(), dim);

    if (result.shape.size() == 1)
        result.shape.emplace(a.shape.begin(), 1);

    result._data = data;
}

} // namespace sk::tensor
