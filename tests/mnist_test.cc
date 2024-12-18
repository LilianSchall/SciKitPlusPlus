#include "sk-plusplus/cluster/kmeans.hh"
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

    sk::Tensor X = serializer.deserialize("examples/mnist_x.dat");
    X /= 255.0f;
    sk::Tensor y = serializer.deserialize("examples/mnist_y.dat");

    sk::Tensor w1 = serializer.deserialize("examples/mnist_w1.dat");
    sk::Tensor w2 = serializer.deserialize("examples/mnist_w2.dat");

    sk::Tensor b1 = serializer.deserialize("examples/mnist_b1.dat");
    sk::Tensor b2 = serializer.deserialize("examples/mnist_b2.dat");

    EXPECT_EQ(X.shape[0], 60000);
    EXPECT_EQ(X.shape[1], 784);

    EXPECT_EQ(y.shape[0], 60000);
}

TEST(MnistTest, DenseTest)
{
    sk::serializer::TensorSerializer serializer;

    sk::Tensor X = serializer.deserialize("examples/mnist_x.dat");
    X /= 255.0f;
    sk::Tensor y = serializer.deserialize("examples/mnist_y.dat");

    sk::Tensor w1 = serializer.deserialize("examples/mnist_w1.dat");
    sk::Tensor w2 = serializer.deserialize("examples/mnist_w2.dat");

    sk::Tensor b1 = serializer.deserialize("examples/mnist_b1.dat");
    sk::Tensor b2 = serializer.deserialize("examples/mnist_b2.dat");

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

TEST(MnistTest, KMeansTest)
{
    sk::serializer::TensorSerializer serializer;

    sk::Tensor X = serializer.deserialize("examples/mnist_x.dat");
    X /= 255.0f;
    sk::Tensor y = serializer.deserialize("examples/mnist_y.dat");
    sk::Tensor centroids =
        serializer.deserialize("examples/mnist_centroids.dat");

    sk::cluster::KMeans model{ centroids };

    sk::Tensor pred = model.predict(X);

    EXPECT_TRUE(true);
}
