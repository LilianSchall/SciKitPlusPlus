#include "tensor.hh"
#include <numeric>
#include <vector>

namespace sk::tensor
{
sk::Tensor global_mean(const sk::Tensor &t)
{
    const std::vector<float> &data = t.as_array();

    float mean =
        std::accumulate(data.cbegin(), data.cend(), 0.0f, std::plus<float>{});
    mean /= data.size();

    return sk::Tensor{ { mean }, { 1 } };
}

sk::Tensor mean(const sk::Tensor &t, int axis)
{
    if (axis == -1 || (axis == 0 && t.shape.size() == 1))
        return global_mean(t);

    assert(
        static_cast<size_t>(axis) < t.shape.size() &&
        "Axis is beyond shape size");

    std::vector<float> means{};

    size_t first_limit = axis == 1 ? t.shape[0] : t.shape[1];
    size_t second_limit = axis == 1 ? t.shape[1] : t.shape[0];

    for (size_t i = 0; i < first_limit; i++)
    {
        float mean = 0;
        for (size_t j = 0; j < second_limit; j++)
        {
            if (axis == 1)
                mean += t(i, j);
            else if (axis == 0)
                mean += t(j, i);
        }

        mean /= second_limit;

        means.push_back(mean);
    }

    return sk::Tensor{ means, { means.size() } };
}
} // namespace sk::tensor
