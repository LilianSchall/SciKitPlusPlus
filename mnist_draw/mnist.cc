#include <sk-plusplus/nn/nn.hh>
#include <sk-plusplus/tensor/tensor.hh>
#include <string>

#include "engine/draw_engine.hh"
#include "sk-plusplus/nn/sequential.hh"
#include "tools/build_model.hh"

int main(void)
{
    std::string title{ "MNIST" };
    DrawEngine draw_engine{ title, 2 * 500, 500 };
    sk::nn::Sequential model;

    build_model(model);

    bool change = false;

    draw_engine.clear();
    draw_engine.present();

    while (draw_engine.run(&change))
    {
        if (change)
        {
            sk::Tensor picture = draw_engine.get_picture();
            sk::Tensor prediction = model.forward(picture);
            draw_engine.prepare_prediction(prediction);
            draw_engine.clear();
            draw_engine.present();
        }

        change = false;
    }

    return 0;
}
