#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>
#include <vector>

TEST(TensorTest, TensorZero)
{
    sk::Tensor t = sk::tensor::zeroes({ 10, 10 });

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 0);
}

TEST(TensorTest, TensorOnes)
{
    sk::Tensor t = sk::tensor::ones({ 10, 10 });

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 1);
}

TEST(TensorTest, TensorRange)
{
    sk::Tensor t = sk::tensor::arange(100);

    for (size_t i = 0; i < t.shape[0]; i++)
            EXPECT_EQ(t(i), i);
}

TEST(TensorTest, TensorReshape)
{
    sk::Tensor t = sk::tensor::ones({10});

    EXPECT_EQ(t.shape.size(), 1);
    EXPECT_EQ(t.shape[0], 10);

    t.reshape({2, 5});

    EXPECT_EQ(t.shape.size(), 2);
    EXPECT_EQ(t.shape[0], 2);
    EXPECT_EQ(t.shape[1], 5);

}

TEST(TensorTest, TensorRangeReshape)
{
    sk::Tensor t = sk::tensor::arange(100);

    t.reshape({10, 10});

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), i * t.shape[1] + j);
}

TEST(TensorTest, TensorTranspose)
{
    sk::Tensor t = sk::tensor::arange(4).reshape({2, 2}).transpose();

    EXPECT_EQ(t(0,0), 0);
    EXPECT_EQ(t(0,1), 2);
    EXPECT_EQ(t(1,0), 1);
    EXPECT_EQ(t(1,1), 3);
}

TEST(TensorTest, TensorArgMax)
{
    sk::Tensor t = sk::tensor::arange(20).reshape({4, 5});

    sk::Tensor coeff_amax = sk::tensor::argmax(t);

    sk::Tensor amax1 = sk::tensor::argmax(t, 1);
    sk::Tensor amax2 = sk::tensor::argmax(t, 0);

    EXPECT_EQ(coeff_amax.shape.size(), 1);
    EXPECT_EQ(coeff_amax.shape[0], 1);
    EXPECT_EQ(coeff_amax(0), 19);


    EXPECT_EQ(amax1.shape.size(), 1);
    EXPECT_EQ(amax1.shape[0], 4);

    for (size_t i = 0; i < amax1.shape[0]; i++)
        EXPECT_EQ(amax1(i), 4);

    EXPECT_EQ(amax2.shape.size(), 1);
    EXPECT_EQ(amax2.shape[0], 5);

    for (size_t i = 0; i < amax2.shape[0]; i++)
        EXPECT_EQ(amax2(i), 3);
}
