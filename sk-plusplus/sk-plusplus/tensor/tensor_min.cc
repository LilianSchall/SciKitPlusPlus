#include "tensor.hh"
#include <numeric>
#include <vector>

namespace sk::tensor
{
sk::Tensor global_min(const sk::Tensor &t)
{
    const std::vector<float> &data = t.as_array();

    float const &(*min_func)(float const &, float const &) = std::min<float>;

    float min = std::accumulate(data.cbegin(), data.cend(), data[0], min_func);

    return sk::Tensor{ { min }, { 1 } };
}

sk::Tensor min(const sk::Tensor &t, int axis)
{
    if (axis == -1 || (axis == 0 && t.shape.size() == 1))
        return global_min(t);

    assert(
        static_cast<size_t>(axis) < t.shape.size() &&
        "Axis is beyond shape size");

    std::vector<float> mins{};

    size_t first_limit = axis == 1 ? t.shape[0] : t.shape[1];
    size_t second_limit = axis == 1 ? t.shape[1] : t.shape[0];

    for (size_t i = 0; i < first_limit; i++)
    {
        float min = axis == 1 ? t(i, 0) : t(0, i);
        for (size_t j = 0; j < second_limit; j++)
        {
            float value;
            if (axis == 1)
                value = t(i, j);
            else // axis == 0
                value = t(j, i);

            if (min > value)
                min = value;
        }

        mins.push_back(min);
    }

    return sk::Tensor{ mins, { mins.size() } };
}
} // namespace sk::tensor
