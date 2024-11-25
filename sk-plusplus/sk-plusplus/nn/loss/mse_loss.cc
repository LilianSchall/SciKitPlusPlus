#include "mse_loss.hh"

namespace sk::nn::loss
{

sk::Tensor MSELoss::forward(sk::Tensor &target, sk::Tensor &prediction)
{
    // TODO: compute MSE
    return Tensor::zeroes({ 1 });
}
} // namespace sk::nn::loss
