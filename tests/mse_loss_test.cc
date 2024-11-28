#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>
#include <sk-plusplus/nn/loss/mse_loss.hh>

TEST(MseTest, PerfectLoss)
{
    sk::nn::loss::MSELoss criterion{};

    sk::Tensor t1{{1, 2, 0}, {3}};
    sk::Tensor t2{{1, 2, 0}, {3}};

    sk::Tensor loss = criterion.forward(t1, t2);

    EXPECT_EQ(loss.shape.size(), 1);
    EXPECT_EQ(loss.shape[0], 1);
    EXPECT_EQ(loss(0), 0);
}

TEST(MseTest, WorstLoss)
{
    sk::nn::loss::MSELoss criterion{};

    sk::Tensor t1{{1, 2, 0}, {3}};
    sk::Tensor t2{{-1, -2, 0}, {3}};

    sk::Tensor loss = criterion.forward(t1, t2);

    EXPECT_EQ(loss.shape.size(), 1);
    EXPECT_EQ(loss.shape[0], 1);
    EXPECT_EQ(loss(0), 20);
}
