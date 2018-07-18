#pragma once
#include <SFML/Graphics/Image.hpp>
#include "RaycasterInterface.hpp"
#include "CudaAutoBuffer.hpp"

class CudaRasterizationParams;

class CudaRaycaster : public RaycasterInterface
{
public:
    CudaRaycaster();
    virtual const char * getRaycasterTechName() const override;
    virtual void rasterize() override;
    virtual void handleKeys() override;
    virtual void setTexture(unsigned texnum, const sf::Image& img) override;
    virtual void setScreenSize(unsigned width, unsigned height) override;
    virtual void downloadImage(sf::Texture& texture) override;
    virtual void loadMap(const sf::Image& img) override;
    virtual CameraExchangeInfo getCameraInfo() const;
    virtual void setCameraInfo(const CameraExchangeInfo& info);

private:
    void setMapSize(unsigned width, unsigned height);
    void setMapTile(unsigned x, unsigned y, unsigned tile);
    unsigned getMapTile(unsigned x, unsigned y) const;

    float m_camposx = 4.5f;
    float m_camposy = 4.5f;
    float m_dirx = -1.f;
    float m_diry = 0.f;
    float m_planex = 0.f;
    float m_planey = 0.66f;

    std::vector<unsigned> m_screen;
    unsigned m_screenwidth;
    unsigned m_screenheight;
    unsigned m_screenpixels;
    std::vector<unsigned> m_map;
    unsigned m_mapwidth;
    unsigned m_mapheight;
    CudaAutoBuffer<unsigned> m_cuda_map;
    CudaAutoBuffer<unsigned> m_cuda_textures;
    CudaAutoBuffer<unsigned> m_cuda_screen;
    CudaAutoBuffer<CudaRasterizationParams> m_cuda_rast_params;

};
