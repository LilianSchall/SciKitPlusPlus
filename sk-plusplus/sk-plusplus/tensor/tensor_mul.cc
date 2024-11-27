#include "tensor.hh"
#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

#include <cblas.h>

namespace sk
{

sk::Tensor sk::Tensor::hadamard_dot(const Tensor &other)
{
    assert(this->shape == other.shape);

    sk::Tensor res = sk::tensor::zeroes(other.shape);

    std::transform(
        this->_data.begin(),
        this->_data.end(),
        other._data.begin(),
        res._data.begin(),
        std::multiplies<float>{});

    return res;
}

sk::Tensor vec_dot(const std::vector<float> &a, const std::vector<float> &b)
{
    assert(a.size() == b.size());
    float res = 0;

    for (size_t i = 0; i < a.size(); i++)
        res += a[i] * b[i];

    return Tensor({ res }, { 1 });
}

sk::Tensor mat_vec_dot(
    const std::vector<float> &a,
    const std::vector<float> &b,
    const size_t a_begin,
    const size_t b_begin,
    const size_t height_a,
    const size_t width_a)
{
    Tensor c = tensor::zeroes({ height_a });

    const std::vector<float> &c_data = c.as_array();

    size_t c_begin = 0;

    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        height_a,
        width_a,
        1.0,
        &a[a_begin],
        height_a,
        &b[b_begin],
        1,
        0.0,
        &const_cast<std::vector<float> &>(c_data)[c_begin],
        1);

    return c;
}

sk::Tensor mat_dot(
    const std::vector<float> &a,
    const std::vector<float> &b,
    const size_t a_begin,
    const size_t b_begin,
    const size_t height_a,
    const size_t width_b,
    const size_t common)
{
    Tensor c = tensor::zeroes({ height_a, width_b });

    const std::vector<float> &c_data = c.as_array();

    size_t c_begin = 0;

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        height_a,
        width_b,
        common,
        1.0,
        &a[a_begin],
        common,
        &b[b_begin],
        width_b,
        0.0,
        &const_cast<std::vector<float> &>(c_data)[c_begin],
        width_b);

    return c;
}

sk::Tensor matmul(Tensor &a, Tensor &b)
{
    size_t dim_a = a.shape.size();
    size_t dim_b = b.shape.size();
    if (dim_a == dim_b)
    {
        if (dim_a == 1)
            return vec_dot(a.as_array(), b.as_array());
        if (dim_a == 2)
        {
            assert(a.shape.end()[-1] == b.shape.end()[-2]);
            return mat_dot(
                a.as_array(),
                b.as_array(),
                0,
                0,
                a.shape[0],
                b.shape[1],
                a.shape[1]);
        }
    }
    if (dim_a == 1 && dim_b == 2)
    {
        a.shape.emplace(a.shape.begin(), 1);

        assert(a.shape.end()[-1] == b.shape.end()[-2]);

        sk::Tensor t = mat_dot(
            a.as_array(),
            b.as_array(),
            0,
            0,
            a.shape[0],
            b.shape[1],
            a.shape[1]);
        a.shape.erase(a.shape.begin());
        t.shape.erase(t.shape.begin());
        return t;
    }
    if (dim_a == 2 && dim_b == 1)
    {
        assert(a.shape.end()[-1] == b.shape.end()[-1]);
        return mat_vec_dot(
            a.as_array(),
            b.as_array(),
            0,
            0,
            a.shape[0],
            a.shape[1]);
    }

    // temporary
    return sk::tensor::zeroes({ 1, 1 });
}

sk::Tensor sk::Tensor::operator*(Tensor &other)
{
    // actually the following line has no sense
    // as we will use broadcasting
    // assert(this->shape == other.shape);

    return matmul(*this, other);
}

sk::Tensor &sk::Tensor::operator*=(Tensor &other)
{
    // actually the following line has no sense
    // as we will use broadcasting
    // assert(this->shape == other.shape);

    sk::Tensor res = matmul(*this, other);

    this->_data = res._data;
    this->shape = res.shape;

    return *this;
}

sk::Tensor operator*(sk::Tensor &lhs, float rhs)
{
    sk::Tensor res = sk::tensor::zeroes(lhs.shape);

    std::transform(
        lhs._data.begin(),
        lhs._data.end(),
        res._data.begin(),
        [rhs](int x) { return x * rhs; });

    return res;
}

sk::Tensor operator*(float lhs, sk::Tensor &rhs)
{
    sk::Tensor res = sk::tensor::zeroes(rhs.shape);

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
