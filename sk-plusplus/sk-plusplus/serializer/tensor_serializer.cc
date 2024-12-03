#include "tensor_serializer.hh"
#include <fstream>
#include <iostream>
#include <istream>
#include <stdexcept>
#include <vector>

namespace sk::serializer
{

template <typename T> T read_value(std::istream &stream, bool *error)
{
    *error = false;
    T value;

    if (!stream.good())
    {
        *error = true;
        return 0;
    }

    // Unsage, but I know what I do.
    stream.read((char *)&value, 4);

    return value;
}

template <typename T> bool write_value(std::ostream &stream, T value)
{
    if (!stream.good())
    {
        return false;
    }

    // Unsage, but I know what I do.
    stream.write((char *)&value, sizeof(value));

    return true;
}

sk::Tensor TensorSerializer::deserialize(const std::string &filepath)
{
    std::ifstream file(filepath, std::ios::in | std::ios::binary);

    if (!file.good())
    {
        throw std::invalid_argument("File does not exist");
    }

    bool error = false;

    unsigned int shape_size = read_value<unsigned int>(file, &error);

    if (error)
    {
        file.close();
        throw std::invalid_argument("Could not read shape size: " + filepath);
    }

    std::vector<size_t> shape;
    size_t nb_elements = 1;

    for (size_t i = 0; i < shape_size && !error; i++)
    {
        unsigned int dim = read_value<unsigned int>(file, &error);
        shape.push_back(dim);
        nb_elements *= dim;
    }


    if (error)
    {
        file.close();
        throw std::invalid_argument("Could not read shape: " + filepath);
    }

    std::vector<float> values;

    for (size_t i = 0; i < nb_elements && !error; i++)
    {
        float value = read_value<float>(file, &error);
        values.push_back(value);
    }

    if (error)
    {
        file.close();
        throw std::invalid_argument("Could not read data: " + filepath);
    }

    return sk::Tensor{ values, shape };
}

void TensorSerializer::serialize(
    const sk::Tensor &t,
    const std::string &filepath)
{
    std::ofstream file(filepath, std::ios::out | std::ios::binary);

    if (!file.good())
    {
        throw std::invalid_argument("File does not exist");
    }

    if (!write_value<unsigned int>(file, t.shape.size()))
    {
        file.close();
        throw std::invalid_argument("File cannot be created");
    }

    const std::vector<size_t> &shape = t.shape;

    for (size_t i = 0; i < shape.size(); i++)
    {
        if (!write_value<unsigned int>(file, shape[i]))
        {
            file.close();
            throw std::invalid_argument("Error during shape writing");
        }
    }

    const std::vector<float> &data = t.as_array();

    for (size_t i = 0; i < data.size(); i++)
    {
        if (!write_value<float>(file, data[i]))
        {
            file.close();
            throw std::invalid_argument("Error during data writing");
        }
    }
}

} // namespace sk::serializer
