#include "sequential.hh"

namespace sk::nn
{

Sequential::Sequential(std::vector<std::shared_ptr<sk::nn::Module>> &layers) :
    _layers(layers)
{
}

sk::Tensor Sequential::forward(sk::Tensor &input)
{
    sk::Tensor output{ input };

    for (std::shared_ptr<sk::nn::Module> &layer : this->_layers)
    {
        output = layer->forward(output);
    }

    return output;
}
} // namespace sk::nn
