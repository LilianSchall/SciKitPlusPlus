#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>
#include <vector>

TEST(TensorSubTest, TensorSubInt)
{
    sk::Tensor t = sk::tensor::zeroes({ 10, 10 });

    for (size_t i = 0; i < 10; i++)
        t -= 1;

    sk::Tensor another_t = t - 1;

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), -10);

    for (size_t i = 0; i < another_t.shape[0]; i++)
        for (size_t j = 0; j < another_t.shape[1]; j++)
            EXPECT_EQ(another_t(i, j), -11);
}

TEST(TensorSubTest, TensorSubFloat)
{
    sk::Tensor t = sk::tensor::zeroes({ 10, 10 });

    for (size_t i = 0; i < 10; i++)
        t -= 1.0f;

    sk::Tensor another_t = t - 1.0f;

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), -10);

    for (size_t i = 0; i < another_t.shape[0]; i++)
        for (size_t j = 0; j < another_t.shape[1]; j++)
            EXPECT_EQ(another_t(i, j), -11);
}

TEST(TensorSubTest, TensorSubTensor)
{
    sk::Tensor t1 = sk::tensor::ones({ 10, 10 });
    sk::Tensor t2 = sk::tensor::ones({ 10, 10 });

    sk::Tensor t3 = t1 - t2;

    for (size_t i = 0; i < t3.shape[0]; i++)
        for (size_t j = 0; j < t3.shape[1]; j++)
            EXPECT_EQ(t3(i, j), 0);
}

TEST(TensorSubTest, TensorSubTensorScalar)
{
    sk::Tensor a = sk::tensor::arange(4).reshape({ 2, 2 });
    sk::Tensor b{ { 1 }, { 1 } };

    sk::Tensor output = a - b;

    for (size_t i = 0; i < a.shape[0]; i++)
        for (size_t j = 0; j < a.shape[1]; j++)
            EXPECT_EQ(output(i, j), a(i, j) - 1);
}
