#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>
#include <vector>

TEST(TensorMulTest, TensorMulInt)
{
    sk::Tensor t = sk::Tensor::ones({ 10, 10 });

    for (size_t i = 0; i < 10; i++)
        t *= 2;

    EXPECT_EQ(t.shape.size(), 2);

    sk::Tensor another_t = t * 2;
    EXPECT_EQ(another_t.shape.size(), 2);

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 1024);

    for (size_t i = 0; i < another_t.shape[0]; i++)
        for (size_t j = 0; j < another_t.shape[1]; j++)
            EXPECT_EQ(another_t(i, j), 2048);
}

TEST(TensorMulTest, TensorMulFloat)
{
    sk::Tensor t = sk::Tensor::ones({ 10, 10 });

    for (size_t i = 0; i < 10; i++)
        t *= 2.0f;

    EXPECT_EQ(t.shape.size(), 2);
    sk::Tensor another_t = t * 2.0f;
    EXPECT_EQ(another_t.shape.size(), 2);

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 1024);

    for (size_t i = 0; i < another_t.shape[0]; i++)
        for (size_t j = 0; j < another_t.shape[1]; j++)
            EXPECT_EQ(another_t(i, j), 2048);
}

TEST(TensorMulTest, VecMulVec)
{
    sk::Tensor t1 = sk::Tensor::ones({ 10 });
    t1 *= 2;
    sk::Tensor t2 = sk::Tensor::ones({ 10 });
    t2 *= 3;

    sk::Tensor t = t1 * t2;

    EXPECT_EQ(t.shape.size(), 1);
    EXPECT_EQ(t.shape[0], 1);

    EXPECT_EQ(t(0), 60);
}

TEST(TensorMulTest, MatrixMulMatrix)
{
    sk::Tensor t1 = sk::Tensor::ones({ 10, 5 });
    t1 *= 2;
    sk::Tensor t2 = sk::Tensor::ones({ 5, 3 });
    t2 *= 3;

    sk::Tensor t = t1 * t2;
    EXPECT_EQ(t.shape.size(), 2);
    EXPECT_EQ(t.shape[0], 10);
    EXPECT_EQ(t.shape[1], 3);

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 30);
}

TEST(TensorMulTest, VecMulMatrix)
{
    sk::Tensor t1 = sk::Tensor::ones({ 10 });
    t1 *= 2;
    sk::Tensor t2 = sk::Tensor::ones({ 10, 10 });
    t2 *= 3;

    sk::Tensor t = t1 * t2;
    EXPECT_EQ(t.shape.size(), 1);
    EXPECT_EQ(t.shape[0], 10);

    for (size_t i = 0; i < t.shape[0]; i++)
            EXPECT_EQ(t(i), 60);
}

TEST(TensorMulTest, MatrixMulVec)
{
    sk::Tensor t1 = sk::Tensor::ones({ 10, 10 });
    t1 *= 2;
    sk::Tensor t2 = sk::Tensor::ones({ 10 });
    t2 *= 3;

    sk::Tensor t = t1 * t2;

    EXPECT_EQ(t.shape.size(), 1);
    EXPECT_EQ(t.shape[0], 10);

    for (size_t i = 0; i < t.shape[0]; i++)
        for (size_t j = 0; j < t.shape[1]; j++)
            EXPECT_EQ(t(i, j), 60);
}
