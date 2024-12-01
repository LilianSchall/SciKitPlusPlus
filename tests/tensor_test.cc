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

    std::cout << t << std::endl;

    t.reshape({10, 10});

    std::cout << t << std::endl;

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
