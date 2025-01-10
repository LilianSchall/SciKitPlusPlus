#include "draw_zone.hh"

DrawZone::DrawZone(SDL_Rect rect, SDL_Renderer *renderer) :
    picture(sk::tensor::zeroes({ MNIST_SIZE, MNIST_SIZE })), _rect(rect)
{
    SDL_Texture *texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_STREAMING,
        MNIST_SIZE,
        MNIST_SIZE);

    if (texture == nullptr)
    {
        throw std::runtime_error("SDL_CreateTexture failed");
    }

    this->_texture =
        std::unique_ptr<SDL_Texture, SDL_TextureDestroyer>(texture);
}

void DrawZone::click(int x, int y)
{
    if (x >= this->_rect.x && x <= this->_rect.x + this->_rect.w &&
        y >= this->_rect.y && y <= this->_rect.y + this->_rect.h)
    {
        int relative_x = (x - this->_rect.x) * MNIST_SIZE / this->_rect.w;
        int relative_y = (y - this->_rect.y) * MNIST_SIZE / this->_rect.h;

        this->picture(relative_y, relative_x) = 1;

        write_in_texture(relative_x, relative_y);
    }
}

void DrawZone::draw(SDL_Renderer *renderer)
{
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderCopy(renderer, this->_texture.get(), NULL, &this->_rect);
    SDL_RenderDrawRect(renderer, &this->_rect);
}

void DrawZone::clear()
{
    this->picture = sk::tensor::zeroes({ MNIST_SIZE, MNIST_SIZE });

    void *pixels;
    int pitch;

    if (SDL_LockTexture(this->_texture.get(), NULL, &pixels, &pitch) != 0)
    {
        throw std::runtime_error("SDL_LockTexture failed");
    }

    uint32_t *pixels32 = (uint32_t *)pixels;

    for (int i = 0; i < MNIST_SIZE; i++)
    {
        for (int j = 0; j < MNIST_SIZE; j++)
        {
            pixels32[i * MNIST_SIZE + j] = 0x00000000;
        }
    }

    SDL_UnlockTexture(this->_texture.get());
}

void DrawZone::write_in_texture(int x, int y)
{
    void *pixels;
    int pitch;

    if (SDL_LockTexture(this->_texture.get(), NULL, &pixels, &pitch) != 0)
    {
        throw std::runtime_error("SDL_LockTexture failed");
    }

    uint32_t *pixels32 = (uint32_t *)pixels;
    pixels32[y * MNIST_SIZE + x] = 0xFFFFFFFF;

    SDL_UnlockTexture(this->_texture.get());
}
