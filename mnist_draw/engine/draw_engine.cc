#include "draw_engine.hh"
#include <SDL2/SDL_ttf.h>
#include <SDL_surface.h>

DrawEngine::DrawEngine(const std::string &title, int width, int height) :
    _width(width), _height(height)
{

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        throw std::runtime_error("SDL_Init failed");
    }

    SDL_Window *window = SDL_CreateWindow(
        title.c_str(),
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        width,
        height,
        SDL_WINDOW_SHOWN);

    if (window == nullptr)
    {
        throw std::runtime_error("SDL_CreateWindow failed");
    }

    SDL_Renderer *renderer =
        SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    if (renderer == nullptr)
    {
        throw std::runtime_error("SDL_CreateRenderer failed");
    }

    if (TTF_Init() != 0)
    {
        throw std::runtime_error("TTF_Init failed");
    }

    TTF_Font *font = TTF_OpenFont("Sans.ttf", 24);

    if (font == nullptr)
    {
        throw std::runtime_error("TTF_OpenFont failed");
    }

    this->_window = std::unique_ptr<SDL_Window, SDL_WindowDestroyer>(window);
    this->_renderer =
        std::unique_ptr<SDL_Renderer, SDL_RendererDestroyer>(renderer);
    this->_font = std::unique_ptr<TTF_Font, TTF_FontDestroyer>(font);

    int draw_zone_width = height / 2;
    int draw_zone_height = height / 2;
    int draw_zone_x = (width - draw_zone_width) / 2;
    int draw_zone_y = (height - draw_zone_height) / 2;

    SDL_Rect rect = { draw_zone_x,
                      draw_zone_y,
                      draw_zone_width,
                      draw_zone_height };

    this->_draw_zone = std::make_unique<DrawZone>(rect, renderer);
    this->_prediction_texture = nullptr;
    this->_prediction_rect = { 0, 0, 0, 0 };
}

DrawEngine::~DrawEngine()
{
    SDL_Quit();
    std::cout << "Successfully quit SDL." << std::endl;
}

void DrawEngine::clear()
{
    SDL_SetRenderDrawColor(this->_renderer.get(), 0, 0, 0, 255);
    SDL_RenderClear(this->_renderer.get());
}

void DrawEngine::present()
{
    this->_draw_zone->draw(this->_renderer.get());
    if (this->_prediction_texture != nullptr)
    {
        SDL_RenderCopy(
            this->_renderer.get(),
            this->_prediction_texture.get(),
            NULL,
            &this->_prediction_rect);
    }
    SDL_RenderPresent(this->_renderer.get());
}

bool DrawEngine::run(bool *change)
{
    SDL_Event event;
    bool is_running = true;

    while (SDL_PollEvent(&event))
    {
        *change = *change || handle_keyboard_event(&event, &is_running);
        *change = *change || handle_sdl_event(&event, &is_running);
        *change = *change || handle_mouse_event(&event);
    }

    return is_running;
}

void DrawEngine::prepare_prediction(const sk::Tensor &prediction)
{
    sk::Tensor pred = sk::tensor::argmax(prediction, 0);

    SDL_Color color = { 255, 255, 255, 255 };

    std::string text;

    for (size_t i = 0; i < prediction.shape[0]; i++)
    {
        text += std::to_string(i) + ": " +
                std::to_string(static_cast<int>(prediction(i) * 100)) + "%";

        if (i != prediction.shape[0] - 1)
        {
            text += "\n";
        }
    }
    SDL_Surface *surface =
        TTF_RenderUTF8_Solid_Wrapped(this->_font.get(), text.c_str(), color, 0);

    TTF_SizeUTF8(
        this->_font.get(),
        text.c_str(),
        &this->_prediction_rect.w,
        &this->_prediction_rect.h);

    this->_prediction_rect.w /= 10;
    this->_prediction_rect.h *= 10;

    this->_prediction_rect.x = this->_width - this->_prediction_rect.w;
    this->_prediction_rect.y = (this->_height - this->_prediction_rect.h) / 2;

    this->_prediction_texture =
        std::unique_ptr<SDL_Texture, SDL_TextureDestroyer>(
            SDL_CreateTextureFromSurface(this->_renderer.get(), surface));
}

sk::Tensor DrawEngine::get_picture() { return this->_draw_zone->picture; }

bool DrawEngine::handle_keyboard_event(SDL_Event *event, bool *is_running)
{
    if (event->type == SDL_KEYDOWN)
    {
        if (event->key.keysym.sym == SDLK_ESCAPE)
        {
            *is_running = false;
            return true;
        }
        if (event->key.keysym.sym == SDLK_c)
        {
            this->_draw_zone->clear();
            return true;
        }
    }

    return false;
}

bool DrawEngine::handle_mouse_event(SDL_Event *event)
{
    if (event->type == SDL_MOUSEMOTION &&
        event->motion.state & SDL_BUTTON_LMASK)
    {
        int x = event->motion.x;
        int y = event->motion.y;

        this->_draw_zone->click(x, y);
        return true;
    }

    return false;
}

bool DrawEngine::handle_sdl_event(SDL_Event *event, bool *is_running)
{
    if (event->type == SDL_QUIT)
    {
        *is_running = false;
        return true;
    }

    return false;
}
