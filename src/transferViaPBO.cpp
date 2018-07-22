#include "transferViaPBO.hpp"
#include "gl_core21.h"
#include <SFML/Graphics/Texture.hpp>
#include <cstring>

void transferViaPBO(unsigned * cudascreen, sf::Texture& texture, unsigned pbo)
{
    sf::Texture::bind(&texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    const int x = static_cast<int>(texture.getSize().x);
    const int y = static_cast<int>(texture.getSize().y);
    const unsigned size = texture.getSize().x * texture.getSize().y * 4;
    auto ptr = (unsigned char*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    if(ptr)
    {
        std::memset(ptr, 0xff, size);
        for(int i = 0; i < size; i += 4)
            ptr[i + 0] = 0x0;

        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        sf::Texture::bind(&texture);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            x, y, 0,
            GL_BGRA,
            GL_UNSIGNED_BYTE,
            0x0
        );
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
