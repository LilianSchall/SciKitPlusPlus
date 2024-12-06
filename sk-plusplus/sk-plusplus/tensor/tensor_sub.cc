#include "tensor.hh"
#include <algorithm>
#include <cassert>
#include <functional>

namespace sk
{

sk::Tensor sk::Tensor::operator-(const Tensor &other)
{
    assert(this->shape == other.shape);

    sk::Tensor res = sk::tensor::zeroes(other.shape);

    std::transform(
        this->_data.begin(),
        this->_data.end(),
        other._data.begin(),
        res._data.begin(),
        std::minus<float>{});

    return res;
}

sk::Tensor &sk::Tensor::operator-=(const Tensor &other)
{
    assert(this->shape == other.shape);

    std::transform(
        this->_data.begin(),
        this->_data.end(),
        other._data.begin(),
        this->_data.begin(),
        std::minus<float>{});

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

sk::Tensor operator-(sk::Tensor &lhs)
{
    return 0.0f - lhs;
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
