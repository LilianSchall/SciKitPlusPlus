#pragma once

#include <cassert>
#include <concepts>
#include <type_traits>
#include <vector>

namespace sk
{

class Tensor
{
  public:
    Tensor(std::vector<float> data, std::vector<size_t> shape);

    static Tensor ones(std::vector<size_t> shape);
    static Tensor zeroes(std::vector<size_t> shape);

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

        for (size_t i = v.size() - 1; i > 0; i--)
        {
            index += metric * v[i];
            metric *= this->shape[i];
        }

        return this->_data[index];
    }

    Tensor operator+(const Tensor &other);
    Tensor &operator+=(const Tensor &other);

    friend Tensor operator+(Tensor &lhs, const float rhs);
    friend Tensor operator+(const float lhs, Tensor &rhs);
    Tensor &operator+=(const float other);

    friend Tensor operator+(Tensor &lhs, const int rhs);
    friend Tensor operator+(const int lhs, Tensor &rhs);
    Tensor &operator+=(const int other);

    Tensor operator*(Tensor &other);
    Tensor &operator*=(Tensor &other);

    friend Tensor operator*(Tensor &lhs, const float rhs);
    friend Tensor operator*(const float lhs, Tensor &rhs);
    Tensor &operator*=(const float other);

    friend Tensor operator*(Tensor &lhs, const int rhs);
    friend Tensor operator*(const int lhs, Tensor &rhs);
    Tensor &operator*=(const int other);

    Tensor hadamard_dot(const Tensor &other);

    friend Tensor vec_dot(const Tensor &a, const Tensor &b);
    friend Tensor mat_dot(
        const Tensor &a,
        const Tensor &b,
        const size_t a_begin,
        const size_t b_begin,
        const size_t m,
        const size_t n,
        const size_t k);
    friend Tensor mat_vec_dot(
        const Tensor &a,
        const Tensor &b,
        const size_t a_begin,
        const size_t b_begin,
        const size_t height_a,
        const size_t width_a);

  public:
    std::vector<size_t> shape;

  private:
    std::vector<float> _data;
};

} // namespace sk
