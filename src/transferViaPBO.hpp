#pragma once

namespace sf {
    class Texture;
}

void transferViaPBO(unsigned * cudascreen, sf::Texture& texture, unsigned pbo);
