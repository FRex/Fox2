#pragma once

#include <SFML/Graphics/Image.hpp>

inline unsigned complementRGB(unsigned color)
{
    const unsigned char r = 255 - ((color >> 24) & 0xff);
    const unsigned char g = 255 - ((color >> 16) & 0xff);
    const unsigned char b = 255 - ((color >> 8) & 0xff);
    const unsigned char a = color & 0xff;
    return (r << 24) + (g << 16) + (b << 8) + a;
}

inline sf::Image makeJorgeImage(unsigned tsize)
{
    sf::Image ret;
    ret.create(tsize, tsize);
    for(int x = 0; x < tsize; ++x)
    {
        for(int y = 0; y < tsize; ++y)
        {
            const int xx = x / 8;
            const int yy = y / 8;
            unsigned c = ((xx + yy) % 2 == 0) ? 0xff00ffff : 0x00007fff;
            if(x - std::abs(static_cast<int>(tsize) - 2 * y) > 0)
                c = complementRGB(c);

            ret.setPixel(x, y, sf::Color(c));
        }//for y
    }//for x
    return ret;
}
