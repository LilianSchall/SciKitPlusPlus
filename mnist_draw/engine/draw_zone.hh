#pragma once

#include "tools/sdl_tools.hh"

#include <memory>
#include <sk-plusplus/tensor/tensor.hh>

#define MNIST_SIZE 28

class DrawZone
{
  public:
    DrawZone() = delete;
    DrawZone(SDL_Rect rect, SDL_Renderer *renderer);
    void click(int x, int y);
    void draw(SDL_Renderer *renderer);
    void clear();

  private:
    void write_in_texture(int x, int y);

  public:
    sk::Tensor picture;

  private:
    SDL_Rect _rect;
    std::unique_ptr<SDL_Texture, SDL_TextureDestroyer> _texture;
};
