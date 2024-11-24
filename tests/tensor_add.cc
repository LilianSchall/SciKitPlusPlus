#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>
#include <vector>

TEST(TensorTest, TensorAddInt)
{
    sk::Tensor t = sk::Tensor::zeroes({ 10, 10 });

    for (size_t i = 0; i < 10; i++)
        t += 1;

    sk::Tensor another_t = t + 1;

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 10);

    for (size_t i = 0; i < another_t.shape[0]; i++)
        for (size_t j = 0; j < another_t.shape[1]; j++)
            EXPECT_EQ(another_t(i, j), 11);
}

TEST(TensorTest, TensorAddFloat)
{
    sk::Tensor t = sk::Tensor::zeroes({ 10, 10 });

    for (size_t i = 0; i < 10; i++)
        t += 1.0f;

    sk::Tensor another_t = t + 1.0f;

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 10);

    for (size_t i = 0; i < another_t.shape[0]; i++)
        for (size_t j = 0; j < another_t.shape[1]; j++)
            EXPECT_EQ(another_t(i, j), 11);
}

TEST(TensorTest, TensorAddTensor)
{
    sk::Tensor t1 = sk::Tensor::ones({ 10, 10 });
    sk::Tensor t2 = sk::Tensor::ones({ 10, 10 });

    sk::Tensor t3 = t1 + t2;

    for (size_t i = 0; i < t3.shape[0]; i++)
        for (size_t j = 0; j < t3.shape[1]; j++)
            EXPECT_EQ(t3(i, j), 2);
}
