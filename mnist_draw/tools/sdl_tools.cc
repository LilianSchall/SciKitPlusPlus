#include "sdl_tools.hh"

void SDL_WindowDestroyer::operator()(SDL_Window *window) const {
    SDL_DestroyWindow(window);
    std::cout << "SDL_Window has been successfully destroyed.\n";
}

void SDL_RendererDestroyer::operator()(SDL_Renderer *renderer) const {
    SDL_DestroyRenderer(renderer);
    std::cout << "SDL_Renderer has been successfully destroyed.\n";
}

void SDL_TextureDestroyer::operator()(SDL_Texture *texture) const {
    SDL_DestroyTexture(texture);
    std::cout << "SDL_Texture has been successfully destroyed.\n";
}

void TTF_FontDestroyer::operator()(TTF_Font *font) const {
    TTF_CloseFont(font);
    TTF_Quit();
    std::cout << "TTF_Font has been successfully destroyed.\n";
}
