#include "tensor.hh"
#include <numeric>
#include <vector>

namespace sk::tensor
{
std::vector<sk::Tensor> global_split(const sk::Tensor &t)
{
    const std::vector<float> &data = t.as_array();

    std::vector<sk::Tensor> output;

    for (float value : data)
        output.push_back(sk::Tensor{ { value }, { 1 } });

    return output;
}

std::vector<sk::Tensor> split(const sk::Tensor &t, int axis)
{
    if (axis == -1 || (axis == 0 && t.shape.size() == 1))
        return global_split(t);

    assert(
        static_cast<size_t>(axis) < t.shape.size() &&
        "Axis is beyond shape size");

    std::vector<sk::Tensor> splits{};

    size_t first_limit = axis == 1 ? t.shape[0] : t.shape[1];
    size_t second_limit = axis == 1 ? t.shape[1] : t.shape[0];

    for (size_t i = 0; i < first_limit; i++)
    {
        sk::Tensor view = sk::tensor::zeroes({second_limit});
        for (size_t j = 0; j < second_limit; j++)
        {
            if (axis == 1)
                view(j) = t(i, j);
            else // axis == 0
                view(j) = t(j, i);
        }
        splits.push_back(view);
    }

    return splits;
}
} // namespace sk::tensor
