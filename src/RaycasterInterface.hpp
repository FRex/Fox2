#pragma once

namespace sf {
    class Texture;
    class Image;
}

class RaycasterInterface
{
public:
    virtual ~RaycasterInterface() = default;
    virtual const char * getRaycasterTechName() const = 0;
    virtual void rasterize() = 0;
    virtual void setScreenSize(unsigned width, unsigned height) = 0;
    virtual void setTexture(unsigned texnum, const sf::Image& img) = 0;
    virtual void downloadImage(sf::Texture& texture) = 0;
    virtual void downloadDepthImage(sf::Texture& texture) = 0;
    virtual void loadMap(const sf::Image& img) = 0;
    virtual void handleKeys() = 0;

private:


};
