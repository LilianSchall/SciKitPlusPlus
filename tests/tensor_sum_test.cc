#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>

TEST(TensorSumTest, TensorSumBasicFloatMatrix)
{
    sk::Tensor a = sk::Tensor{ { 0.25, 2.25, 2.25, 0.25 }, { 4 } };

    a.reshape({ 2, 2 });

    sk::Tensor sum = sk::tensor::sum(a);

    EXPECT_EQ(sum.shape.size(), 1);
    EXPECT_EQ(sum.shape[0], 1);
    EXPECT_EQ(sum(0), 5);
}

TEST(TensorSumTest, TensorSumAll)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({ 2, 2 });

    sk::Tensor sum = sk::tensor::sum(a);

    EXPECT_EQ(sum.shape.size(), 1);
    EXPECT_EQ(sum.shape[0], 1);
    EXPECT_EQ(sum(0), 10);
}

TEST(TensorSumTest, TensorSumAxisZero)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({ 2, 2 });

    sk::Tensor sum = sk::tensor::sum(a, 0);

    EXPECT_EQ(sum.shape.size(), 1);
    EXPECT_EQ(sum.shape[0], 2);
    EXPECT_EQ(sum(0), 4);
    EXPECT_EQ(sum(1), 6);
}

TEST(TensorSumTest, TensorSumAxisOne)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({ 2, 2 });

    sk::Tensor sum = sk::tensor::sum(a, 1);

    EXPECT_EQ(sum.shape.size(), 1);
    EXPECT_EQ(sum.shape[0], 2);
    EXPECT_EQ(sum(0), 3);
    EXPECT_EQ(sum(1), 7);
}
