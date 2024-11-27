#include "mse_loss.hh"

namespace sk::nn::loss
{

sk::Tensor MSELoss::forward(sk::Tensor &target, sk::Tensor &prediction)
{
    sk::Tensor loss = target - prediction;

    loss *= loss;

    return loss;
}
} // namespace sk::nn::loss
