#include "sk-plusplus/nn/logistic_regression.hh"
#include "sk-plusplus/serializer/tensor_serializer.hh"
#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>

TEST(LogisticRegressionTest, LoadModel)
{

    sk::serializer::TensorSerializer serializer;

    sk::Tensor X = serializer.deserialize("examples/x.dat");

    EXPECT_EQ(X.shape.size(), 2);
    EXPECT_EQ(X.shape[0], 150);
    EXPECT_EQ(X.shape[1], 4);

    sk::Tensor y = serializer.deserialize("examples/y.dat");

    EXPECT_EQ(y.shape.size(), 1);
    EXPECT_EQ(y.shape[0], 150);


    sk::Tensor w = serializer.deserialize("examples/weights.dat");

    EXPECT_EQ(w.shape.size(), 2);
    EXPECT_EQ(w.shape[0], 3);
    EXPECT_EQ(w.shape[1], 4);

    std::cout << "Before transpose:\n" << w << std::endl;
    
    w.transpose();

    std::cout << "After transpose:\n" << w << std::endl;

    EXPECT_EQ(w.shape.size(), 2);
    EXPECT_EQ(w.shape[0], 4);
    EXPECT_EQ(w.shape[1], 3);

    sk::Tensor b = serializer.deserialize("examples/biases.dat");

    EXPECT_EQ(b.shape.size(), 1);
    EXPECT_EQ(b.shape[0], 3);

    sk::nn::LogisticRegression clf{w, b};

    sk::Tensor pred = clf.forward(X);

    std::cout << "pred: " << pred << std::endl;
}
