#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>

TEST(TensorSplitTest, TensorSplitBasicFloatMatrix)
{
    sk::Tensor a = sk::Tensor{ { 0.25, 2.25, 2.25, 0.25 }, { 4 } };

    a.reshape({ 2, 2 });

    std::vector<sk::Tensor> split = sk::tensor::split(a);

    EXPECT_EQ(split.size(), 4);
    EXPECT_EQ(split[0](0), 0.25f);
    EXPECT_EQ(split[1](0), 2.25f);
    EXPECT_EQ(split[2](0), 2.25f);
    EXPECT_EQ(split[3](0), 0.25f);
}

TEST(TensorSplitTest, TensorSplitAll)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    std::vector<sk::Tensor> split = sk::tensor::split(a);

    EXPECT_EQ(split.size(), 4);
    EXPECT_EQ(split[0](0), 1);
    EXPECT_EQ(split[1](0), 2);
    EXPECT_EQ(split[2](0), 3);
    EXPECT_EQ(split[3](0), 4);
}

TEST(TensorSplitTest, TensorSplitAxisZero)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    std::vector<sk::Tensor> split = sk::tensor::split(a, 0);

    EXPECT_EQ(split.size(), 2);
    EXPECT_EQ(split[0](0), 1);
    EXPECT_EQ(split[0](1), 3);
    EXPECT_EQ(split[1](0), 2);
    EXPECT_EQ(split[1](1), 4);
}

TEST(TensorSplitTest, TensorSplitAxisOne)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({2, 2});

    std::vector<sk::Tensor> split = sk::tensor::split(a, 1);

    EXPECT_EQ(split.size(), 2);
    EXPECT_EQ(split[0](0), 1);
    EXPECT_EQ(split[0](1), 2);
    EXPECT_EQ(split[1](0), 3);
    EXPECT_EQ(split[1](1), 4);
}
