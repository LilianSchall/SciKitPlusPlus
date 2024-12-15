#include "tensor.hh"
#include <vector>

namespace sk::tensor
{
sk::Tensor global_argmax(const sk::Tensor &t)
{
    const std::vector<float> &data = t.as_array();
    size_t best_index = 0;
    for (size_t i = 0; i < data.size(); i++)
    {
        if (data[best_index] < data[i])
            best_index = i;
    }

    return sk::Tensor{ { static_cast<float>(best_index) }, { 1 } };
}

sk::Tensor argmax(const sk::Tensor &t, int axis)
{
    if (axis == -1 || (axis == 0 && t.shape.size() == 1))
        return global_argmax(t);

    assert(
        static_cast<size_t>(axis) < t.shape.size() &&
        "Axis is beyond shape size");

    std::vector<float> best{};

    size_t first_limit = axis == 1 ? t.shape[0] : t.shape[1];
    size_t second_limit = axis == 1 ? t.shape[1] : t.shape[0];

    for (size_t i = 0; i < first_limit; i++)
    {
        size_t best_index = 0;
        for (size_t j = 0; j < second_limit; j++)
        {
            if ((axis == 1 && t(i, j) > t(i, best_index)) ||
                (axis == 0 && t(j, i) > t(best_index, i)))
                best_index = j;
        }

        best.push_back(best_index);
    }

    return sk::Tensor{ best, { best.size() } };
}

} // namespace sk::tensor
