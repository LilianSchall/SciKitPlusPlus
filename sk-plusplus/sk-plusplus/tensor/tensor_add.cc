#include "tensor.hh"
#include <algorithm>
#include <cassert>
#include <functional>

namespace sk
{

sk::Tensor sk::Tensor::operator+(const Tensor &other)
{
    sk::Tensor res = sk::tensor::zeroes(other.shape);

    if (this->shape == other.shape)
    {
        std::transform(
            this->_data.begin(),
            this->_data.end(),
            other._data.begin(),
            res._data.begin(),
            std::plus<float>{});

        return res;
    }

    // temporary, just to be able to broadcast row vector with matrix
    if (this->shape[-1] == other.shape[-1] && this->shape.size() == 2 &&
        other.shape.size() == 1)
    {
        for (size_t i = 0; i < this->shape[0]; i++)
        {
            std::transform(
                this->_data.begin() + i * this->shape[1],
                this->_data.begin() + (i + 1) * this->shape[1],
                other._data.begin(),
                res._data.begin() + i * res.shape[1],
                std::plus<float>{});
        }
        return res;
    }

    assert(false && "Cannot broadcast tensors for addition");
}

sk::Tensor &sk::Tensor::operator+=(const Tensor &other)
{
    if (this->shape == other.shape)
    {
        std::transform(
            this->_data.begin(),
            this->_data.end(),
            other._data.begin(),
            this->_data.begin(),
            std::plus<float>{});

        return *this;
    }

    // temporary, just to be able to broadcast row vector with matrix
    if (this->shape[-1] == other.shape[-1] && this->shape.size() == 2 &&
        other.shape.size() == 1)
    {
        for (size_t i = 0; i < this->shape[0]; i++)
        {
            std::transform(
                this->_data.begin() + i * this->shape[1],
                this->_data.begin() + (i + 1) * this->shape[1],
                other._data.begin(),
                this->_data.begin() + i * this->shape[1],
                std::plus<float>{});
        }

        return *this;
    }

    assert(false && "Cannot broadcast tensors for addition");
}

sk::Tensor operator+(sk::Tensor &lhs, float rhs)
{
    sk::Tensor res = sk::tensor::zeroes(lhs.shape);

    std::transform(
        lhs._data.begin(),
        lhs._data.end(),
        res._data.begin(),
        [rhs](float x) { return x + rhs; });

    return res;
}

sk::Tensor operator+(float lhs, sk::Tensor &rhs)
{
    sk::Tensor res = sk::tensor::zeroes(rhs.shape);

    std::transform(
        rhs._data.begin(),
        rhs._data.end(),
        res._data.begin(),
        [lhs](float x) { return x + lhs; });

    return res;
}

sk::Tensor &sk::Tensor::operator+=(float other)
{
    std::transform(
        this->_data.begin(),
        this->_data.end(),
        this->_data.begin(),
        [other](float x) { return x + other; });

    return *this;
}

sk::Tensor operator+(sk::Tensor &lhs, int rhs)
{
    return lhs + static_cast<float>(rhs);
}

sk::Tensor operator+(const int lhs, sk::Tensor &rhs)
{
    return rhs + static_cast<float>(lhs);
}

sk::Tensor &sk::Tensor::operator+=(int other)
{
    *this += static_cast<float>(other);
    return *this;
}
} // namespace sk
