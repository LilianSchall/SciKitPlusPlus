#include "sk-plusplus/nn/linear.hh"
#include "sk-plusplus/nn/module.hh"
#include <gtest/gtest.h>

#include <memory>
#include <sk-plusplus/nn/nn.hh>
#include <sk-plusplus/serializer/tensor_serializer.hh>
#include <sk-plusplus/tensor/tensor.hh>
#include <vector>

TEST(MnistTest, DatasetTest)
{
    sk::serializer::TensorSerializer serializer;

    sk::Tensor X = serializer.deserialize("examples/x_mnist.dat");
    X /= 255.0f;
    sk::Tensor y = serializer.deserialize("examples/y_mnist.dat");

    sk::Tensor w1 = serializer.deserialize("examples/w1.dat");
    sk::Tensor w2 = serializer.deserialize("examples/w2.dat");

    sk::Tensor b1 = serializer.deserialize("examples/b1.dat");
    sk::Tensor b2 = serializer.deserialize("examples/b2.dat");

    EXPECT_EQ(X.shape[0], 60000);
    EXPECT_EQ(X.shape[1], 784);

    EXPECT_EQ(y.shape[0], 60000);
}

TEST(MnistTest, AccuracyTest)
{
    sk::serializer::TensorSerializer serializer;

    sk::Tensor X = serializer.deserialize("examples/x_mnist.dat");
    X /= 255.0f;
    sk::Tensor y = serializer.deserialize("examples/y_mnist.dat");

    sk::Tensor w1 = serializer.deserialize("examples/w1.dat");
    sk::Tensor w2 = serializer.deserialize("examples/w2.dat");

    sk::Tensor b1 = serializer.deserialize("examples/b1.dat");
    sk::Tensor b2 = serializer.deserialize("examples/b2.dat");

    std::vector<std::shared_ptr<sk::nn::Module>> layers{};
    layers.push_back(std::make_shared<sk::nn::Linear>(w1, b1));
    layers.push_back(std::make_shared<sk::nn::Relu>());
    layers.push_back(std::make_shared<sk::nn::Linear>(w2, b2));
    layers.push_back(std::make_shared<sk::nn::Sigmoid>());

    sk::nn::Sequential model{ layers };

    sk::Tensor pred = model.forward(X);

    pred = sk::tensor::argmax(pred, 1);

    sk::Tensor count = pred == y;

    float s = sk::tensor::sum(count)(0);
    float accuracy = s / y.shape[0];

    std::cout << "Accuracy: " << accuracy << std::endl;

    EXPECT_GE(accuracy, 0.97f);
}
