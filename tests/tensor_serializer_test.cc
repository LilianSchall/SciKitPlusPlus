#include <gtest/gtest.h>

#include <sk-plusplus/tensor/tensor.hh>
#include <sk-plusplus/serializer/tensor_serializer.hh>

TEST(TensorSerializerTest, SerializeDeserialize)
{
    sk::Tensor t = sk::tensor::arange(100);

    t.reshape({10, 10});

    sk::serializer::TensorSerializer serializer;

    const std::string filepath = "test.dat";

    serializer.serialize(t, filepath);

    sk::Tensor result = serializer.deserialize(filepath);

    EXPECT_EQ(result.shape.size(), 2);
    EXPECT_EQ(result.shape[0], 10);
    EXPECT_EQ(result.shape[1], 10);

    for (size_t i = 0; i < result.shape[0]; i++)
        for (size_t j = 0; j < result.shape[1]; j++)
            EXPECT_EQ(result(i, j), i * result.shape[1] + j);
}
