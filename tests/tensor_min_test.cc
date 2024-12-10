#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>

TEST(TensorMinTest, TensorMinBasicFloatMatrix)
{
    sk::Tensor a = sk::Tensor{ { 0.25, 2.25, 2.25, 0.25 }, { 4 } };

    a.reshape({ 2, 2 });

    sk::Tensor min = sk::tensor::min(a);

    EXPECT_EQ(min.shape.size(), 1);
    EXPECT_EQ(min.shape[0], 1);
    EXPECT_EQ(min(0), 0.25f);
}

TEST(TensorMinTest, TensorMinAll)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor min = sk::tensor::min(a);

    EXPECT_EQ(min.shape.size(), 1);
    EXPECT_EQ(min.shape[0], 1);
    EXPECT_EQ(min(0), 1);
}

TEST(TensorMinTest, TensorMinAxisZero)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor min = sk::tensor::min(a, 0);

    EXPECT_EQ(min.shape.size(), 1);
    EXPECT_EQ(min.shape[0], 2);
    EXPECT_EQ(min(0), 1);
    EXPECT_EQ(min(1), 2);
}

TEST(TensorMinTest, TensorMinAxisOne)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor min = sk::tensor::min(a, 1);

    EXPECT_EQ(min.shape.size(), 1);
    EXPECT_EQ(min.shape[0], 2);
    EXPECT_EQ(min(0), 1);
    EXPECT_EQ(min(1), 3);
}
