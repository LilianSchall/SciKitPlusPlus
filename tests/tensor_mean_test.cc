#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>

TEST(TensorMeanTest, TensorMeanBasicFloatMatrix)
{
    sk::Tensor a = sk::Tensor{ { 0.25, 2.25, 2.25, 0.25 }, { 4 } };

    a.reshape({ 2, 2 });

    sk::Tensor mean = sk::tensor::mean(a);

    EXPECT_EQ(mean.shape.size(), 1);
    EXPECT_EQ(mean.shape[0], 1);
    EXPECT_EQ(mean(0), 1.25f);
}

TEST(TensorMeanTest, TensorMeanAll)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor mean = sk::tensor::mean(a);

    EXPECT_EQ(mean.shape.size(), 1);
    EXPECT_EQ(mean.shape[0], 1);
    EXPECT_EQ(mean(0), 2.5f);
}

TEST(TensorMeanTest, TensorMeanAxisZero)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor mean = sk::tensor::mean(a, 0);

    EXPECT_EQ(mean.shape.size(), 1);
    EXPECT_EQ(mean.shape[0], 2);
    EXPECT_EQ(mean(0), 2);
    EXPECT_EQ(mean(1), 3);
}

TEST(TensorMeanTest, TensorMeanAxisOne)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    sk::Tensor mean = sk::tensor::mean(a, 1);

    EXPECT_EQ(mean.shape.size(), 1);
    EXPECT_EQ(mean.shape[0], 2);
    EXPECT_EQ(mean(0), 1.5f);
    EXPECT_EQ(mean(1), 3.5f);
}
