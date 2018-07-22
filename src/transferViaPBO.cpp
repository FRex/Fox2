#include "transferViaPBO.hpp"
#include "gl_core21.h"
#include <SFML/Graphics/Texture.hpp>

void transferViaPBO(sf::Texture& texture, unsigned pbo)
{
    /*
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_screenpixels * 4, 0x0, GL_STREAM_DRAW);

    void * ptr = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    if(ptr)
    {
        std::memset(ptr, 0xff, m_screenpixels * 4);
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        sf::Texture::bind(&texture);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            texture.getSize().x,
            texture.getSize().y,
            0,
            GL_BGRA,
            GL_UNSIGNED_INT,
            0x0
        );
    }
    else
    {
        std::printf("WAT?!@?!\n");
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    */
}
