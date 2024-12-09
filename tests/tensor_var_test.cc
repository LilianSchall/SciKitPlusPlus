#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>

TEST(TensorVarTest, TensorVarAll)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor var = sk::tensor::var(a);

    EXPECT_EQ(var.shape.size(), 1);
    EXPECT_EQ(var.shape[0], 1);
    EXPECT_EQ(var(0), 1.25f);
}

TEST(TensorVarTest, TensorVarAxisZero)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor var = sk::tensor::var(a, 0);

    EXPECT_EQ(var.shape.size(), 1);
    EXPECT_EQ(var.shape[0], 2);
    EXPECT_EQ(var(0), 1);
    EXPECT_EQ(var(1), 1);
}

TEST(TensorVarTest, TensorVarAxisOne)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor var = sk::tensor::var(a, 1);

    EXPECT_EQ(var.shape.size(), 1);
    EXPECT_EQ(var.shape[0], 2);
    EXPECT_EQ(var(0), 0.25f);
    EXPECT_EQ(var(1), 0.25f);
}
