#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>
#include <vector>

TEST(TensorTest, TensorZero)
{
    sk::Tensor t = sk::Tensor::zeroes({ 10, 10 });

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 0);
}

TEST(TensorTest, TensorOnes)
{
    sk::Tensor t = sk::Tensor::ones({ 10, 10 });

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 1);
}
