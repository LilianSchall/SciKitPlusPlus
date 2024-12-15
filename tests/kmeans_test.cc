#include "sk-plusplus/cluster/kmeans.hh"
#include "sk-plusplus/serializer/tensor_serializer.hh"
#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>

TEST(KmeansTest, LoadModel)
{

    sk::serializer::TensorSerializer serializer;

    sk::Tensor X = serializer.deserialize("examples/iris_x.dat");

    EXPECT_EQ(X.shape.size(), 2);
    EXPECT_EQ(X.shape[0], 150);
    EXPECT_EQ(X.shape[1], 4);

    sk::Tensor true_pred = serializer.deserialize("examples/kmeans_pred.dat");

    EXPECT_EQ(true_pred.shape.size(), 1);
    EXPECT_EQ(true_pred.shape[0], 150);

    sk::Tensor centroids = serializer.deserialize("examples/kmeans_centroids.dat");

    EXPECT_EQ(centroids.shape.size(), 2);
    EXPECT_EQ(centroids.shape[0], 3);
    EXPECT_EQ(centroids.shape[1], 4);

    sk::cluster::KMeans clf{centroids};

    sk::Tensor pred = clf.predict(X);

    EXPECT_EQ(pred.shape.size(), true_pred.shape.size());
    EXPECT_EQ(pred.shape[0], true_pred.shape[0]);

    sk::Tensor count = pred == true_pred;

    float s = sk::tensor::sum(count)(0);

    EXPECT_GE(s / true_pred.shape[0], 1);
}
