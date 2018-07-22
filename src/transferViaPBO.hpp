#pragma once

namespace sf {
    class Texture;
}

void transferViaPBO(sf::Texture& texture, unsigned pbo);
