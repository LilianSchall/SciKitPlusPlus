#pragma once

#include <vector>

namespace sk
{

class Tensor
{
  public:
    Tensor(std::vector<float> data, std::vector<size_t> shape);

    std::vector<size_t> create_iter(void);

    static Tensor ones(std::vector<size_t> shape);
    static Tensor zeroes(std::vector<size_t> shape);

    float& operator[](const std::vector<size_t>& indices);

    Tensor operator+(const Tensor &other);
    Tensor &operator+=(const Tensor &other);

    friend Tensor operator+(Tensor &lhs, const float rhs);
    friend Tensor operator+(const float lhs, Tensor &rhs);
    Tensor &operator+=(const float other);

    friend Tensor operator+(Tensor &lhs, const int rhs);
    friend Tensor operator+(const int lhs, Tensor &rhs);
    Tensor &operator+=(const int other);

    Tensor operator*(const Tensor &other);
    Tensor &operator*=(const Tensor &other);

    friend Tensor operator*(Tensor &lhs, const float rhs);
    friend Tensor operator*(const float lhs, Tensor &rhs);
    Tensor &operator*=(const float other);

    friend Tensor operator*(Tensor &lhs, const int rhs);
    friend Tensor operator*(const int lhs, Tensor &rhs);
    Tensor &operator*=(const int other);

    Tensor hadamard_dot(const Tensor &other);

  public:
    std::vector<size_t> shape;

  private:
    std::vector<float> _data;
};

} // namespace sk
