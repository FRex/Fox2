#pragma once
#include <SFML/Graphics/Image.hpp>
#include "RaycasterInterface.hpp"

namespace fox {

class FoxRaycaster : public RaycasterInterface
{
public:
    FoxRaycaster();
    virtual const char * getRaycasterTechName() const override;
    virtual void rasterize() override;
    virtual void handleKeys() override;
    virtual void setTexture(unsigned texnum, const sf::Image& img) override;
    virtual void setScreenSize(unsigned width, unsigned height) override;
    virtual void downloadImage(sf::Texture& texture) override;
    virtual void loadMap(const sf::Image& img) override;
    virtual CameraExchangeInfo getCameraInfo() const;
    virtual void setCameraInfo(const CameraExchangeInfo& info);
    void setName(const std::string& name);
    virtual float getRasterTime() override;

private:
    void setMapSize(unsigned width, unsigned height);
    void setMapTile(unsigned x, unsigned y, unsigned tile);
    unsigned * getTexture(unsigned num);
    const unsigned * getTexture(unsigned num) const;
    unsigned screenPixelIndex(unsigned x, unsigned y);
    unsigned getMapTile(unsigned x, unsigned y) const;

    float m_camposx = 4.5f;
    float m_camposy = 4.5f;
    float m_dirx = -1.f;
    float m_diry = 0.f;
    float m_planex = 0.f;
    float m_planey = 0.66f;

    std::vector<unsigned> m_screen;
    std::vector<unsigned> m_textures;
    unsigned m_screenwidth;
    unsigned m_screenheight;
    unsigned m_screenpixels;
    std::vector<unsigned> m_map;
    unsigned m_mapwidth;
    unsigned m_mapheight;
    std::string m_name;
    float m_rastertime = 0.f;

};

}//fox
