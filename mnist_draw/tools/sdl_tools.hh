#pragma once

#include <SDL2/SDL.h>
#include <SDL2/SDL_render.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <iostream>
#include <exception>

struct SDL_WindowDestroyer
{
    void operator()(SDL_Window *window) const;
};

struct SDL_RendererDestroyer
{
    void operator()(SDL_Renderer *renderer) const;
};

struct SDL_TextureDestroyer
{
    void operator()(SDL_Texture *texture) const;
};

struct TTF_FontDestroyer
{
    void operator()(TTF_Font *font) const;
};
