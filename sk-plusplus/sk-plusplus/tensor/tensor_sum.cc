#include "tensor.hh"
#include <numeric>
#include <vector>

namespace sk::tensor
{
sk::Tensor global_sum(const sk::Tensor &t)
{
    const std::vector<float> &data = t.as_array();

    const float sum =
        std::accumulate(data.cbegin(), data.cend(), 0.0f, std::plus<float>{});

    return sk::Tensor{ { sum }, { 1 } };
}

sk::Tensor sum(const sk::Tensor &t, int axis)
{
    if (axis == -1 || (axis == 0 && t.shape.size() == 1))
        return global_sum(t);

    assert(
        static_cast<size_t>(axis) < t.shape.size() &&
        "Axis is beyond shape size");

    std::vector<float> sums{};

    size_t first_limit = axis == 1 ? t.shape[0] : t.shape[1];
    size_t second_limit = axis == 1 ? t.shape[1] : t.shape[0];

    for (size_t i = 0; i < first_limit; i++)
    {
        float sum = 0;
        for (size_t j = 0; j < second_limit; j++)
        {
            if (axis == 1)
                sum += t(i, j);
            else if (axis == 0)
                sum += t(j, i);
        }

        sums.push_back(sum);
    }

    return sk::Tensor{ sums, { sums.size() } };
}
} // namespace sk::tensor
