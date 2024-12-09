#include "tensor.hh"
#include <vector>

namespace sk::tensor
{
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
