#pragma once

#include "engine/draw_zone.hh"
#include "tools/sdl_tools.hh"

#include <SDL2/SDL.h>
#include <memory>
#include <sk-plusplus/tensor/tensor.hh>

class DrawEngine
{
  public:
    DrawEngine() = delete;
    DrawEngine(const std::string &title, int width, int height);
    ~DrawEngine();

    void prepare_prediction(const sk::Tensor &prediction);
    void clear();
    void present();
    bool run(bool *change);
    sk::Tensor get_picture();

  private:
    bool handle_keyboard_event(SDL_Event *event, bool *is_running);
    bool handle_sdl_event(SDL_Event *event, bool *is_running);
    bool handle_mouse_event(SDL_Event *event);

  private:
    std::unique_ptr<SDL_Window, SDL_WindowDestroyer> _window;
    std::unique_ptr<SDL_Renderer, SDL_RendererDestroyer> _renderer;
    int _width;
    int _height;
    std::unique_ptr<DrawZone> _draw_zone;
    std::unique_ptr<SDL_Texture, SDL_TextureDestroyer> _prediction_texture;
    std::unique_ptr<TTF_Font, TTF_FontDestroyer> _font;
    SDL_Rect _prediction_rect;
};
