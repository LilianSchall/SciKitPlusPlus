#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>
#include <vector>

TEST(TensorTest, TensorZero)
{
    sk::Tensor t = sk::Tensor::zeroes({ 10, 10 });

    std::vector<size_t> iter = t.create_iter();

    for (; iter[0] < t.shape[0]; iter[0]++)
        for (; iter[1] < t.shape[1]; iter[1]++)
            EXPECT_EQ(t[iter], 0);
}

TEST(TensorTest, TensorOnes)
{
    sk::Tensor t = sk::Tensor::ones({ 10, 10 });

    std::vector<size_t> iter = t.create_iter();

    for (; iter[0] < t.shape[0]; iter[0]++)
        for (; iter[1] < t.shape[1]; iter[1]++)
            EXPECT_EQ(t[iter], 1);
}

TEST(TensorTest, TensorAddInt)
{
    sk::Tensor t = sk::Tensor::zeroes({ 10, 10 });

    for (size_t i = 0; i < 10; i++)
        t += 1;

    sk::Tensor another_t = t + 1;

    std::vector<size_t> iter = t.create_iter();

    for (; iter[0] < t.shape[0]; iter[0]++)
        for (; iter[1] < t.shape[1]; iter[1]++)
            EXPECT_EQ(t[iter], 10);

    iter = another_t.create_iter();

    for (; iter[0] < another_t.shape[0]; iter[0]++)
        for (; iter[1] < another_t.shape[1]; iter[1]++)
            EXPECT_EQ(another_t[iter], 11);
}

TEST(TensorTest, TensorAddFloat)
{
    sk::Tensor t = sk::Tensor::zeroes({ 10, 10 });

    for (size_t i = 0; i < 10; i++)
        t += 1.0f;

    sk::Tensor another_t = t + 1.0f;

    std::vector<size_t> iter = t.create_iter();

    for (; iter[0] < t.shape[0]; iter[0]++)
        for (; iter[1] < t.shape[1]; iter[1]++)
            EXPECT_EQ(t[iter], 10);

    iter = another_t.create_iter();

    for (; iter[0] < another_t.shape[0]; iter[0]++)
        for (; iter[1] < another_t.shape[1]; iter[1]++)
            EXPECT_EQ(another_t[iter], 11);
}

TEST(TensorTest, TensorAddTensor)
{
    sk::Tensor t1 = sk::Tensor::ones({ 10, 10 });
    sk::Tensor t2 = sk::Tensor::ones({ 10, 10 });

    sk::Tensor t3 = t1 + t2;

    std::vector<size_t> iter = t3.create_iter();

    for (; iter[0] < t3.shape[0]; iter[0]++)
        for (; iter[1] < t3.shape[1]; iter[1]++)
            EXPECT_EQ(t3[iter], 2);
}
