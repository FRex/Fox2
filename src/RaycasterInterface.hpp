#pragma once

namespace sf {
    class Texture;
    class Image;
}

class CameraExchangeInfo
{
public:
    float camposx = 4.5f;
    float camposy = 4.5f;
    float dirx = -1.f;
    float diry = 0.f;
    float planex = 0.f;
    float planey = 0.66f;

};

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
    virtual CameraExchangeInfo getCameraInfo() const = 0;
    virtual void setCameraInfo(const CameraExchangeInfo& info) = 0;

private:


};
