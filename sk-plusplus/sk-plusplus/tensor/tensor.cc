#include "tensor.hh"
#include <memory>
#include <numeric>

namespace sk
{
Tensor::Tensor(std::vector<float> data, std::vector<size_t> shape) :
    shape(shape), _data(data)
{
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

size_t print(
    std::ostream &out,
    const std::vector<float> &data,
    const std::vector<size_t> &shape,
    size_t shape_index,
    size_t data_index)
{
    if (shape_index >= shape.size())
        return data_index;

    out << "[ ";

    for (size_t i = 0; i < shape[shape_index]; i++)
    {
        if (shape_index == shape.size() - 1)
            out << data[data_index++] << " ";
        else
            data_index = print(out, data, shape, shape_index + 1, data_index);
    }
    out << "]\n";

    return data_index;
}

void pretty_print(const sk::Tensor &t, std::ostream &out)
{
    print(out, t.as_array(), t.shape, 0, 0);
}

} // namespace sk::tensor

std::ostream &operator<<(std::ostream &out, const sk::Tensor &t)
{
    sk::tensor::pretty_print(t, out);
    return out;
}
