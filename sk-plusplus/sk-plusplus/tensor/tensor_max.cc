#include "tensor.hh"
#include <numeric>
#include <vector>

namespace sk::tensor
{
sk::Tensor global_max(const sk::Tensor &t)
{
    const std::vector<float> &data = t.as_array();

    float const &(*max_func)(float const &, float const &) = std::max<float>;

    float max = std::accumulate(data.cbegin(), data.cend(), data[0], max_func);

    return sk::Tensor{ { max }, { 1 } };
}

sk::Tensor max(const sk::Tensor &t, int axis)
{
    if (axis == -1 || (axis == 0 && t.shape.size() == 1))
        return global_max(t);

    assert(
        static_cast<size_t>(axis) < t.shape.size() &&
        "Axis is beyond shape size");

    std::vector<float> maxs{};

    size_t first_limit = axis == 1 ? t.shape[0] : t.shape[1];
    size_t second_limit = axis == 1 ? t.shape[1] : t.shape[0];

    for (size_t i = 0; i < first_limit; i++)
    {
        float max = axis == 1 ? t(i, 0) : t(0, i);
        for (size_t j = 0; j < second_limit; j++)
        {
            float value;
            if (axis == 1)
                value = t(i, j);
            else // axis == 0
                value = t(j, i);

            if (max < value)
                max = value;
        }

        maxs.push_back(max);
    }

    return sk::Tensor{ maxs, { maxs.size() } };
}
} // namespace sk::tensor
