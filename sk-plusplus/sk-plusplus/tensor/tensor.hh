#pragma once

#include <cassert>
#include <concepts>
#include <functional>
#include <type_traits>
#include <vector>

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

    friend Tensor operator-(Tensor &lhs, const float rhs);
    friend Tensor operator-(const float lhs, Tensor &rhs);
    Tensor &operator-=(const float other);

    friend Tensor operator-(Tensor &lhs, const int rhs);
    friend Tensor operator-(const int lhs, Tensor &rhs);
    Tensor &operator-=(const int other);

    Tensor operator*(Tensor &other);
    Tensor &operator*=(Tensor &other);

    friend Tensor operator*(Tensor &lhs, const float rhs);
    friend Tensor operator*(const float lhs, Tensor &rhs);
    Tensor &operator*=(const float other);

    friend Tensor operator*(Tensor &lhs, const int rhs);
    friend Tensor operator*(const int lhs, Tensor &rhs);
    Tensor &operator*=(const int other);

    Tensor hadamard_dot(const Tensor &other);

    const std::vector<float> &as_array() const;

    Tensor &reshape(std::vector<size_t> shape);
    Tensor &map(std::function<float(float)> func);

  public:
    std::vector<size_t> shape;

  private:
    std::vector<float> _data;
};

} // namespace sk

namespace sk::tensor
{
sk::Tensor ones(std::vector<size_t> shape);
sk::Tensor zeroes(std::vector<size_t> shape);
sk::Tensor arange(int max, int min = 0, int step = 1);
} // namespace sk::tensor
