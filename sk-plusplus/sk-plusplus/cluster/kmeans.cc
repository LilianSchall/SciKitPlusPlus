#include "kmeans.hh"
#include "sk-plusplus/tensor/tensor.hh"
#include <vector>

namespace sk::cluster
{
KMeans::KMeans(sk::Tensor &centroids): centroids_(centroids){}

sk::Tensor KMeans::predict(sk::Tensor &input) {

    std::vector<sk::Tensor> points = sk::tensor::split(input, 1);

    sk::Tensor output = sk::tensor::zeroes({input.shape[0]});

    for (size_t i = 0; i < output.shape[0]; i++)
    {
        sk::Tensor distances = sk::tensor::sqrt(this->centroids_ * points[i].transpose());

        output(i) = sk::tensor::argmin(distances)(0);
    }

    return output;
}

} // namespace sk::cluster
