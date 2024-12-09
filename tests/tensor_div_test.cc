#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>
#include <vector>

TEST(TensorDivTest, TensorDivInt)
{
    sk::Tensor t = sk::tensor::ones({ 10, 10 });

    t *= 10;

    sk::Tensor another_t = t / 2;

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 10);

    for (size_t i = 0; i < another_t.shape[0]; i++)
        for (size_t j = 0; j < another_t.shape[1]; j++)
            EXPECT_EQ(another_t(i, j), 5);
}

TEST(TensorDivTest, TensorDivFloat)
{
    sk::Tensor t = sk::tensor::ones({ 10, 10 });

    t *= 10.0f;

    sk::Tensor another_t = t / 2.0f;

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 10);

    for (size_t i = 0; i < another_t.shape[0]; i++)
        for (size_t j = 0; j < another_t.shape[1]; j++)
            EXPECT_EQ(another_t(i, j), 5);
}

TEST(TensorDivTest, TensorDivTensor)
{
    sk::Tensor t1 = sk::tensor::ones({ 10, 10 });
    sk::Tensor t2 = sk::tensor::ones({ 10, 10 });

    t2 *= 2;

    sk::Tensor t3 = t1 / t2;

    for (size_t i = 0; i < t3.shape[0]; i++)
        for (size_t j = 0; j < t3.shape[1]; j++)
            EXPECT_EQ(t3(i, j), 0.5f);
}

TEST(TensorDivTest, TensorBroadcastDivTensor)
{
    sk::Tensor t1 = sk::tensor::ones({ 10, 10 });
    sk::Tensor t2 = sk::tensor::arange(10);

    t2 += 1;

    sk::Tensor t3 = t1 / t2;

    for (size_t i = 0; i < t3.shape[0]; i++)
        for (size_t j = 0; j < t3.shape[1]; j++)
            EXPECT_EQ(t3(i, j), 1.0f / (j + 1));
}

TEST(TensorDivTest, TensorDivTensorScalar)
{
    sk::Tensor a = sk::tensor::arange(4).reshape({ 2, 2 });
    sk::Tensor b{ { 5 }, { 1 } };

    sk::Tensor output = a / b;

    for (size_t i = 0; i < a.shape[0]; i++)
        for (size_t j = 0; j < a.shape[1]; j++)
            EXPECT_EQ(output(i, j), a(i, j) / 5);
}
