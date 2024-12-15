#pragma once

#include <cassert>
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>

// Forward declaration
namespace sk
{
class Tensor;
}

namespace sk::tensor
{
sk::Tensor add(const sk::Tensor &a, const sk::Tensor &b);
void add(const sk::Tensor &a, const sk::Tensor &b, sk::Tensor &result);

sk::Tensor sub(const sk::Tensor &a, const sk::Tensor &b);
void sub(const sk::Tensor &a, const sk::Tensor &b, sk::Tensor &result);

sk::Tensor mul(sk::Tensor &a, sk::Tensor &b);
void mul(sk::Tensor &a, sk::Tensor &b, sk::Tensor &result);

sk::Tensor div(const sk::Tensor &a, const sk::Tensor &b);
void div(const sk::Tensor &a, const sk::Tensor &b, sk::Tensor &result);

sk::Tensor transpose(const sk::Tensor &a);
void transpose(const sk::Tensor &a, sk::Tensor &result);

sk::Tensor ones(std::vector<size_t> shape);
sk::Tensor zeroes(std::vector<size_t> shape);
sk::Tensor arange(int max, int min = 0, int step = 1);
sk::Tensor argmax(const sk::Tensor &t, int axis = -1);
sk::Tensor argmin(const sk::Tensor &t, int axis = -1);
sk::Tensor sum(const sk::Tensor &t, int axis = -1);
sk::Tensor mean(const sk::Tensor &t, int axis = -1);
sk::Tensor min(const sk::Tensor &t, int axis = -1);
sk::Tensor max(const sk::Tensor &t, int axis = -1);
std::vector<sk::Tensor> split(const sk::Tensor &t, int axis = -1);
sk::Tensor var(sk::Tensor &t, int axis = -1, int ddof = 0);
sk::Tensor sqrt(const sk::Tensor &t);
sk::Tensor std(sk::Tensor &t, int axis = -1, int ddof = 0);
void pretty_print(const sk::Tensor &t, std::ostream &out = std::cout);

} // namespace sk::tensor

std::ostream &operator<<(std::ostream &out, const sk::Tensor &t);

namespace sk
{

class Tensor
{
  public:
    Tensor(std::vector<float> data, std::vector<size_t> shape);

    template <typename... Ints> float &operator()(Ints... indices)
    {

        static_assert(
            (std::conjunction_v<std::is_integral<Ints>...>),
            "All indices must be integers");
        size_t index = 0;

        std::vector<size_t> v{ static_cast<size_t>(indices)... };

        assert(
            this->shape.size() == v.size() &&
            "shape does not match number of given indices");

        size_t metric = 1;

        for (size_t i = v.size(); i > 0; i--)
        {
            index += metric * v[i - 1];
            metric *= this->shape[i - 1];
        }

        return this->_data[index];
    }

    template <typename... Ints> float operator()(Ints... indices) const
    {
        static_assert(
            (std::conjunction_v<std::is_integral<Ints>...>),
            "All indices must be integers");
        size_t index = 0;

        std::vector<size_t> v{ static_cast<size_t>(indices)... };

        assert(
            this->shape.size() == v.size() &&
            "shape does not match number of given indices");

        size_t metric = 1;

        for (size_t i = v.size(); i > 0; i--)
        {
            index += metric * v[i - 1];
            metric *= this->shape[i - 1];
        }

        return this->_data[index];
    }

    // add operators
    Tensor operator+(const Tensor &other);
    Tensor &operator+=(const Tensor &other);

    friend Tensor operator+(Tensor &lhs, const float rhs);
    friend Tensor operator+(const float lhs, Tensor &rhs);
    Tensor &operator+=(const float other);

    friend Tensor operator+(Tensor &lhs, const int rhs);
    friend Tensor operator+(const int lhs, Tensor &rhs);
    Tensor &operator+=(const int other);

    Tensor operator-(const Tensor &other);
    Tensor &operator-=(const Tensor &other);

    // sub operators
    friend Tensor operator-(Tensor &lhs);

    friend Tensor operator-(Tensor &lhs, const float rhs);
    friend Tensor operator-(const float lhs, Tensor &rhs);
    Tensor &operator-=(const float other);

    friend Tensor operator-(Tensor &lhs, const int rhs);
    friend Tensor operator-(const int lhs, Tensor &rhs);
    Tensor &operator-=(const int other);

    // mul operators
    Tensor operator*(Tensor &other);
    Tensor &operator*=(Tensor &other);

    friend Tensor operator*(Tensor &lhs, const float rhs);
    friend Tensor operator*(const float lhs, Tensor &rhs);
    Tensor &operator*=(const float other);

    friend Tensor operator*(Tensor &lhs, const int rhs);
    friend Tensor operator*(const int lhs, Tensor &rhs);
    Tensor &operator*=(const int other);

    Tensor hadamard_dot(const Tensor &other);

    // div operators
    Tensor operator/(const Tensor &other);
    Tensor &operator/=(const Tensor &other);

    friend Tensor operator/(Tensor &lhs, const float rhs);
    friend Tensor operator/(const float lhs, Tensor &rhs);
    Tensor &operator/=(const float other);

    friend Tensor operator/(Tensor &lhs, const int rhs);
    friend Tensor operator/(const int lhs, Tensor &rhs);
    Tensor &operator/=(const int other);

    // ---- sk::tensor functions
    // ---- arithmetic operations
    friend Tensor sk::tensor::add(const sk::Tensor &a, const sk::Tensor &b);
    friend void sk::tensor::add(
        const sk::Tensor &a,
        const sk::Tensor &b,
        sk::Tensor &result);

    friend Tensor sk::tensor::sub(const sk::Tensor &a, const sk::Tensor &b);
    friend void sk::tensor::sub(
        const sk::Tensor &a,
        const sk::Tensor &b,
        sk::Tensor &result);

    friend Tensor sk::tensor::mul(sk::Tensor &a, sk::Tensor &b);
    friend void
    sk::tensor::mul(sk::Tensor &a, sk::Tensor &b, sk::Tensor &result);

    friend Tensor sk::tensor::div(const sk::Tensor &a, const sk::Tensor &b);
    friend void sk::tensor::div(
        const sk::Tensor &a,
        const sk::Tensor &b,
        sk::Tensor &result);

    // ---- structural operations
    friend sk::Tensor sk::tensor::transpose(const sk::Tensor &a);
    friend void sk::tensor::transpose(const sk::Tensor &a, sk::Tensor &result);

    friend Tensor operator==(const Tensor &lhs, const Tensor &rhs);

    const std::vector<float> &as_array() const;

    Tensor &reshape(std::vector<size_t> shape);
    Tensor &map(std::function<float(float)> func);
    Tensor &transpose(void);

  public:
    std::vector<size_t> shape;

  private:
    std::vector<float> _data;
};

} // namespace sk
