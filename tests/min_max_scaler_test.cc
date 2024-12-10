#include <gtest/gtest.h>

#include <sk-plusplus/scaler/scaler.hh>
#include <sk-plusplus/tensor/tensor.hh>

TEST(MinMaxScalerTest, ScaleMatrix)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({ 2, 2 });

    sk::scaler::MinMaxScaler scaler{ a };

    sk::Tensor scaled = scaler.transform(a);

    EXPECT_EQ(scaled.shape.size(), 2);
    EXPECT_EQ(scaled.shape[0], 2);
    EXPECT_EQ(scaled.shape[1], 2);
    EXPECT_EQ(scaled(0, 0), 0);
    EXPECT_EQ(scaled(0, 1), 0);
    EXPECT_EQ(scaled(1, 0), 1);
    EXPECT_EQ(scaled(1, 1), 1);
}

TEST(MinMaxScalerTest, ScaleUnscaleMatrix)
{
    sk::Tensor a = sk::tensor::arange(4);
    a += 1;
    a.reshape({ 2, 2 });

    sk::scaler::MinMaxScaler scaler{ a };

    sk::Tensor scaled = scaler.transform(a);
    sk::Tensor unscaled = scaler.inverse_transform(scaled);

    EXPECT_EQ(unscaled.shape.size(), 2);
    EXPECT_EQ(unscaled.shape[0], 2);
    EXPECT_EQ(unscaled.shape[1], 2);
    EXPECT_EQ(unscaled(0, 0), a(0, 0));
    EXPECT_EQ(unscaled(0, 1), a(0, 1));
    EXPECT_EQ(unscaled(1, 0), a(1, 0));
    EXPECT_EQ(unscaled(1, 1), a(1, 1));
}
