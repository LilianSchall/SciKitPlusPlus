#include "build_model.hh"

void build_model(sk::nn::Sequential &model)
{
    sk::serializer::TensorSerializer serializer;

    sk::Tensor w1 = serializer.deserialize("examples/mnist_w1.dat");
    sk::Tensor w2 = serializer.deserialize("examples/mnist_w2.dat");
    sk::Tensor b1 = serializer.deserialize("examples/mnist_b1.dat");
    sk::Tensor b2 = serializer.deserialize("examples/mnist_b2.dat");

    model.add_layer(std::make_shared<sk::nn::Flatten>());
    model.add_layer(std::make_shared<sk::nn::Linear>(w1, b1));
    model.add_layer(std::make_shared<sk::nn::Relu>());
    model.add_layer(std::make_shared<sk::nn::Linear>(w2, b2));
    model.add_layer(std::make_shared<sk::nn::Softmax>());
}
