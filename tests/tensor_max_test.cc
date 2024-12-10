#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>

TEST(TensorMaxTest, TensorMaxBasicFloatMatrix)
{
    sk::Tensor a = sk::Tensor{ { 0.25, 2.25, 2.25, 0.25 }, { 4 } };

    a.reshape({ 2, 2 });

    sk::Tensor max = sk::tensor::max(a);

    EXPECT_EQ(max.shape.size(), 1);
    EXPECT_EQ(max.shape[0], 1);
    EXPECT_EQ(max(0), 2.25f);
}

TEST(TensorMaxTest, TensorMaxAll)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor max = sk::tensor::max(a);

    EXPECT_EQ(max.shape.size(), 1);
    EXPECT_EQ(max.shape[0], 1);
    EXPECT_EQ(max(0), 4);
}

TEST(TensorMaxTest, TensorMaxAxisZero)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor max = sk::tensor::max(a, 0);

    EXPECT_EQ(max.shape.size(), 1);
    EXPECT_EQ(max.shape[0], 2);
    EXPECT_EQ(max(0), 3);
    EXPECT_EQ(max(1), 4);
}

TEST(TensorMaxTest, TensorMaxAxisOne)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor max = sk::tensor::max(a, 1);

    EXPECT_EQ(max.shape.size(), 1);
    EXPECT_EQ(max.shape[0], 2);
    EXPECT_EQ(max(0), 2);
    EXPECT_EQ(max(1), 4);
}
