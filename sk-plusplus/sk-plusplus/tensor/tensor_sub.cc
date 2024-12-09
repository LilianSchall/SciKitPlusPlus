#include "tensor.hh"
#include <algorithm>
#include <cassert>
#include <functional>

namespace sk
{
// unary operator
sk::Tensor operator-(sk::Tensor &lhs) { return 0.0f - lhs; }

sk::Tensor sk::Tensor::operator-(const Tensor &other)
{
    return sk::tensor::sub(*this, other);
}

sk::Tensor &sk::Tensor::operator-=(const Tensor &other)
{
    sk::tensor::sub(*this, other, *this);

    return *this;
}

sk::Tensor operator-(sk::Tensor &lhs, float rhs)
{
    sk::Tensor res = sk::tensor::zeroes(lhs.shape);

    std::transform(
        lhs._data.begin(),
        lhs._data.end(),
        res._data.begin(),
        [rhs](float x) { return x - rhs; });

    return res;
}

sk::Tensor operator-(float lhs, sk::Tensor &rhs)
{
    sk::Tensor res = sk::tensor::zeroes(rhs.shape);

    std::transform(
        rhs._data.begin(),
        rhs._data.end(),
        res._data.begin(),
        [lhs](float x) { return lhs - x; });

    return res;
}

sk::Tensor &sk::Tensor::operator-=(float other)
{
    std::transform(
        this->_data.begin(),
        this->_data.end(),
        this->_data.begin(),
        [other](float x) { return x - other; });

    return *this;
}

sk::Tensor operator-(sk::Tensor &lhs, int rhs)
{
    return lhs - static_cast<float>(rhs);
}

sk::Tensor operator-(const int lhs, sk::Tensor &rhs)
{
    return static_cast<float>(lhs) - rhs;
}

sk::Tensor &sk::Tensor::operator-=(int other)
{
    *this -= static_cast<float>(other);
    return *this;
}
} // namespace sk

namespace sk::tensor
{

void sub(const sk::Tensor &a, const sk::Tensor &b, sk::Tensor &result)
{
    if (a.shape == b.shape)
    {
        std::transform(
            a._data.begin(),
            a._data.end(),
            b._data.begin(),
            result._data.begin(),
            std::minus<float>{});
        return;
    }

    // temporary, just to be able to broadcast row vector with matrix
    if (a.shape.size() == 2 && b.shape.size() == 1 && a.shape[1] == b.shape[0])
    {
        for (size_t i = 0; i < a.shape[0]; i++)
        {
            std::transform(
                a._data.begin() + i * a.shape[1],
                a._data.begin() + (i + 1) * a.shape[1],
                b._data.begin(),
                result._data.begin() + i * result.shape[1],
                std::minus<float>{});
        }
        return;
    }

    if (b.shape.size() == 1 && b.shape[0] == 1)
    {
        float value = b(0);

        std::transform(
            a._data.begin(),
            a._data.end(),
            result._data.begin(),
            [value](float x) { return x - value; });
        return;
    }

    assert(false && "Cannot broadcast tensors for subition");
}

sk::Tensor sub(const sk::Tensor &a, const sk::Tensor &b)
{
    sk::Tensor res = sk::tensor::zeroes(a.shape);
    sk::tensor::sub(a, b, res);

    return res;
}

} // namespace sk::tensor
