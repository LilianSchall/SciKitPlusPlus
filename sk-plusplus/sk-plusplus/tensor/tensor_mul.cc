#include "tensor.hh"
#include <algorithm>
#include <cassert>
#include <functional>

namespace sk
{

sk::Tensor sk::Tensor::hadamard_dot(const Tensor &other)
{
    assert(this->shape == other.shape);

    sk::Tensor res = sk::Tensor::zeroes(other.shape);

    std::transform(
        this->_data.begin(),
        this->_data.end(),
        other._data.begin(),
        res._data.begin(),
        std::multiplies<float>{});

    return res;
}

sk::Tensor vec_dot(const Tensor &a, const Tensor &b)
{
    assert(a.shape[0] == b.shape[0]);
    float res = 0;

    for (size_t i = 0; i < a.shape[0]; i++)
        res += a._data[i] * b._data[i];

    return Tensor({res}, {});
}

sk::Tensor mat_dot(const Tensor &a, const Tensor &b)
{

}

sk::Tensor matmul(const Tensor &a, const Tensor &b)
{
    size_t dim_a = a.shape.size();
    size_t dim_b = b.shape.size();
    if (dim_a == dim_b)
    {
        if (dim_a == 1)
            return vec_dot(a, b);
        if (dim_a == 2)
            return mat_dot(a, b);
    }
}

sk::Tensor sk::Tensor::operator*(const Tensor &other)
{
    // TODO: implement matmul
    assert(this->shape == other.shape);

    return matmul(*this, other);
}

sk::Tensor &sk::Tensor::operator*=(const Tensor &other)
{
    // TODO: implement matmul
    assert(this->shape == other.shape);

    sk::Tensor res = matmul(*this, other);

    this->_data = res._data;
    this->shape = res.shape;

    return *this;
}

sk::Tensor operator*(sk::Tensor &lhs, float rhs)
{
    sk::Tensor res = sk::Tensor::zeroes(lhs.shape);

    std::transform(
        lhs._data.begin(),
        lhs._data.end(),
        res._data.begin(),
        [rhs](int x) { return x * rhs; });

    return res;
}

sk::Tensor operator*(float lhs, sk::Tensor &rhs)
{
    sk::Tensor res = sk::Tensor::zeroes(rhs.shape);

    std::transform(
        rhs._data.begin(),
        rhs._data.end(),
        res._data.begin(),
        [lhs](int x) { return x * lhs; });

    return res;
}

sk::Tensor &sk::Tensor::operator*=(float other)
{
    std::transform(
        this->_data.begin(),
        this->_data.end(),
        this->_data.begin(),
        [other](int x) { return x * other; });

    return *this;
}

sk::Tensor operator*(sk::Tensor &lhs, int rhs)
{
    return lhs * static_cast<float>(rhs);
}

sk::Tensor operator*(const int lhs, sk::Tensor &rhs)
{
    return rhs * static_cast<float>(lhs);
}

sk::Tensor &sk::Tensor::operator*=(int other)
{
    *this *= static_cast<float>(other);
    return *this;
}
} // namespace sk
