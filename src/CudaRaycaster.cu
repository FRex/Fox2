#include "CudaRaycaster.hpp"
#include <cmath>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Graphics/Texture.hpp>
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "checkCudaCall.hpp"

/*
Original raycasting code from tutorials at: http://lodev.org/cgtutor/index.html

Copyright (c) 2004-2007, Lode Vandevenne

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

const unsigned kTextureSize = 64u;
const unsigned kTexturePixels = kTextureSize * kTextureSize;

inline unsigned texturePixelIndex(unsigned x, unsigned y)
{
    return x + kTextureSize * y;
}

__device__ unsigned cuda_texturePixelIndex(unsigned x, unsigned y)
{
    return x + kTextureSize * y;
}

__device__ unsigned cuda_halveRGB(unsigned color)
{
    const unsigned char r = ((color >> 24) & 0xff) / 2;
    const unsigned char g = ((color >> 16) & 0xff) / 2;
    const unsigned char b = ((color >> 8) & 0xff) / 2;
    const unsigned char a = color & 0xff;
    return (r << 24) + (g << 16) + (b << 8) + a;
}

unsigned complementRGB(unsigned color)
{
    const unsigned char r = 255 - ((color >> 24) & 0xff);
    const unsigned char g = 255 - ((color >> 16) & 0xff);
    const unsigned char b = 255 - ((color >> 8) & 0xff);
    const unsigned char a = color & 0xff;
    return (r << 24) + (g << 16) + (b << 8) + a;
}

class CudaRasterizationParams
{
public:
    float camposx;
    float camposy;
    float dirx;
    float diry;
    float planex;
    float planey;
    int screenwidth;
    int screenheight;
    unsigned * screen;
    unsigned mapwidth;
    unsigned mapheight;
    unsigned * map;
    unsigned * textures;
    unsigned texturecount;
};

static sf::Image makeJorgeImage()
{
    sf::Image ret;
    ret.create(kTextureSize, kTextureSize);
    for(int x = 0; x < kTextureSize; ++x)
    {
        for(int y = 0; y < kTextureSize; ++y)
        {
            const int xx = x / 8;
            const int yy = y / 8;
            unsigned c = ((xx + yy) % 2 == 0) ? 0xff00ffff : 0x00007fff;
            if(x - std::abs(static_cast<int>(kTextureSize) - 2 * y) > 0)
                c = complementRGB(c);

            ret.setPixel(x, y, sf::Color(c));
        }//for y
    }//for x
    return ret;
}

CudaRaycaster::CudaRaycaster()
{
    setScreenSize(800u, 600u);
    setMapSize(10u, 10u);
    setTexture(0u, makeJorgeImage());
    m_cuda_rast_params.resize(1u);
}

const char * CudaRaycaster::getRaycasterTechName() const
{
    return "cuda";
}

inline unsigned getMapTile(const CudaRasterizationParams * params, unsigned x, unsigned y)
{
    if(x < params->mapwidth && y < params->mapheight)
        return params->map[x + params->mapwidth * y];

    return 1u;
}

__device__ unsigned cuda_getMapTile(const CudaRasterizationParams * params, unsigned x, unsigned y)
{
    if(x < params->mapwidth && y < params->mapheight)
        return params->map[x + params->mapwidth * y];

    return 1u;
}

__device__ unsigned cuda_screenPixelIndex(const CudaRasterizationParams * params, unsigned x, unsigned y)
{
    return x + params->screenwidth * y;
}

__device__ const unsigned * cuda_getTexture(const CudaRasterizationParams * params, unsigned num)
{
    if(num < params->texturecount)
        return params->textures + num * kTexturePixels;

    return params->textures; //jorge
}

__global__ void clearScreen(unsigned * screen, unsigned width, unsigned height, unsigned color)
{
    unsigned * row = screen + blockIdx.x * width;
    for(int i = 0; i < width; ++i)
        row[i] = color;
}

__global__ void cuda_rasterizeColumn(const CudaRasterizationParams * params)
{
    const int x = blockIdx.x;
    if(x >= params->screenwidth)
        return;

    //calculate ray position and direction
    const float camerax = 2.f * x / static_cast<float>(params->screenwidth) - 1.f; //x-coordinate in camera space
    const float rayposx = params->camposx;
    const float rayposy = params->camposy;
    const float raydirx = params->dirx + params->planex * camerax;
    const float raydiry = params->diry + params->planey * camerax;

    //which box of the map we're in
    int mapx = static_cast<int>(rayposx);
    int mapy = static_cast<int>(rayposy);

    //length of ray from current position to next x or y-side
    float sidedistx;
    float sidedisty;

    //length of ray from one x or y-side to next x or y-side
    const float deltadistx = std::sqrt(1 + (raydiry * raydiry) / (raydirx * raydirx));
    const float deltadisty = std::sqrt(1 + (raydirx * raydirx) / (raydiry * raydiry));
    float perpwalldist;

    //what direction to step in x or y-direction (either +1 or -1)
    int stepx;
    int stepy;

    int hit = 0; //was there a wall hit?
    int side; //was a NS or a EW wall hit?
              //calculate step and initial sideDist
    if(raydirx < 0)
    {
        stepx = -1;
        sidedistx = (rayposx - mapx) * deltadistx;
    }
    else
    {
        stepx = 1;
        sidedistx = (mapx + 1.f - rayposx) * deltadistx;
    }
    if(raydiry < 0)
    {
        stepy = -1;
        sidedisty = (rayposy - mapy) * deltadisty;
    }
    else
    {
        stepy = 1;
        sidedisty = (mapy + 1.f - rayposy) * deltadisty;
    }

    //perform DDA
    while(hit == 0)
    {
        //jump to next map square, OR in x-direction, OR in y-direction
        if(sidedistx < sidedisty)
        {
            sidedistx += deltadistx;
            mapx += stepx;
            side = 0;
        }
        else
        {
            sidedisty += deltadisty;
            mapy += stepy;
            side = 1;
        }
        //Check if ray has hit a wall
        hit = cuda_getMapTile(params, mapx, mapy) > 0u;
    }

    //Calculate distance projected on camera direction (oblique distance will give fisheye effect!)
    if(side == 0)
        perpwalldist = (mapx - rayposx + (1 - stepx) / 2) / raydirx;
    else
        perpwalldist = (mapy - rayposy + (1 - stepy) / 2) / raydiry;

    //Calculate height of line to draw on screen
    const int lineheight = static_cast<int>(params->screenheight / perpwalldist);

    //calculate lowest and highest pixel to fill in current stripe
    int drawstart = -lineheight / 2 + params->screenheight / 2;
    if(drawstart < 0)
        drawstart = 0;

    int drawend = lineheight / 2 + params->screenheight / 2;
    if(drawend >= params->screenheight)
        drawend = params->screenheight - 1;

    //choose wall color
    if(cuda_getMapTile(params, mapx, mapy) > 0)
    {
        float wallx;
        if(side == 0)
            wallx = rayposy + perpwalldist * raydiry;
        else
            wallx = rayposx + perpwalldist * raydirx;

        wallx -= std::floor((wallx));

        int texx = static_cast<int>(wallx * static_cast<float>(kTextureSize));
        if(side == 0 && raydirx > 0)
            texx = kTextureSize - texx - 1;

        if(side == 1 && raydiry < 0)
            texx = kTextureSize - texx - 1;

        for(int y = drawstart; y < drawend; y++)
        {
            const int d = y * 256 - params->screenheight * 128 + lineheight * 128;  //256 and 128 factors to avoid floats
            const int texy = ((d * kTextureSize) / lineheight) / 256;
            const unsigned * tex0 = cuda_getTexture(params, cuda_getMapTile(params, mapx, mapy));

            unsigned color = tex0[cuda_texturePixelIndex(texx, texy)];
            if(side == 1)
                color = cuda_halveRGB(color);

            params->screen[cuda_screenPixelIndex(params, x, y)] = color;
        }//for y

         //FLOOR CASTING:
        float floorxwall, floorywall;

        //4 different wall directions possible
        if(side == 0 && raydirx > 0.f)
        {
            floorxwall = static_cast<float>(mapx);
            floorywall = static_cast<float>(mapy + wallx);
        }
        else if(side == 0 && raydirx < 0.f)
        {
            floorxwall = mapx + 1.f;
            floorywall = mapy + wallx;
        }
        else if(side == 1 && raydiry > 0.f)
        {
            floorxwall = mapx + wallx;
            floorywall = static_cast<float>(mapy);
        }
        else
        {
            floorxwall = mapx + wallx;
            floorywall = mapy + 1.f;
        }

        float distwall, distplayer;
        distwall = perpwalldist;
        distplayer = 0.f;

        if(drawend < 0)
            drawend = params->screenheight; //becomes < 0 when the integer overflows

        //draw the floor from drawEnd to the bottom of the screen
        const unsigned * origfloortex = cuda_getTexture(params, 2u);
        const unsigned * origceiltex = cuda_getTexture(params, 0u);
        for(int y = drawend + 1; y < params->screenheight; ++y)
        {
            const float currentdist = params->screenheight / (2.f * y - params->screenheight); //you could make a small lookup table for this instead
            const float weight = (currentdist - distplayer) / (distwall - distplayer);
            const float currentfloorx = weight * floorxwall + (1.f - weight) * params->camposx;
            const float currentfloory = weight * floorywall + (1.f - weight) * params->camposy;
            const int floortexx = static_cast<int>(currentfloorx * kTextureSize) % kTextureSize;
            const int floortexy = static_cast<int>(currentfloory * kTextureSize) % kTextureSize;

            const unsigned * floortex = origfloortex;
            const unsigned * ceiltex = origceiltex;
            if((static_cast<int>(currentfloorx) + static_cast<int>(currentfloory)) % 2)
            {
                const unsigned * tmp = floortex;
                floortex = ceiltex;
                ceiltex = tmp;
            }

            //floor and symmetrical ceiling
            params->screen[cuda_screenPixelIndex(params, x, y)] = floortex[cuda_texturePixelIndex(floortexx, floortexy)];
            params->screen[cuda_screenPixelIndex(params, x, params->screenheight - y)] = ceiltex[cuda_texturePixelIndex(floortexx, floortexy)];
        }
    }//if world map > 0

    for(int y = 0; y < params->screenheight; ++y)
    {
        const unsigned n = params->screen[cuda_screenPixelIndex(params, x, y)];
        params->screen[cuda_screenPixelIndex(params, x, y)] =
            ((n >> 24) & 0xff)    |
            ((n << 8) & 0xff0000) |
            ((n >> 8) & 0xff00)   |
            ((n << 24) & 0xff000000);
    }
}

void CudaRaycaster::rasterize()
{
    CudaRasterizationParams params;
    params.camposx = m_camposx;
    params.camposy = m_camposy;
    params.dirx = m_dirx;
    params.diry = m_diry;
    params.planex = m_planex;
    params.planey = m_planey;
    params.screen = m_cuda_screen.ptr();
    params.screenheight = m_screenheight;
    params.screenwidth = m_screenwidth;
    params.mapwidth = m_mapwidth;
    params.mapheight = m_mapheight;
    params.map = m_cuda_map.ptr();
    params.textures = m_cuda_textures.ptr();
    params.texturecount = m_cuda_textures.size() / kTexturePixels;

    checkCudaCall(cudaMemcpy(m_cuda_rast_params.ptr(), &params, sizeof(CudaRasterizationParams), cudaMemcpyHostToDevice));
    clearScreen << <m_screenheight, 1 >> > (m_cuda_screen.ptr(), m_screenwidth, m_screenheight, 0x7f7f7fff);
    cuda_rasterizeColumn << <m_screenwidth, 1 >> > (m_cuda_rast_params.ptr());
    checkCudaCall(cudaMemcpy(m_screen.data(), m_cuda_screen.ptr(), m_screenpixels * 4u, cudaMemcpyDeviceToHost));
}

void CudaRaycaster::handleKeys()
{
    //speed modifiers
    const float boost = 2.f;
    const float movespeed = boost * (5.f / 60.f);
    const float rotspeed = boost * (3.f / 60.f);

    //move forward if no wall in front of you
    if(sf::Keyboard::isKeyPressed(sf::Keyboard::W))
    {
        if(getMapTile(m_camposx + m_dirx * movespeed, m_camposy) == 0)
            m_camposx += m_dirx * movespeed;

        if(getMapTile(m_camposx, m_camposy + m_diry * movespeed) == 0)
            m_camposy += m_diry * movespeed;
    }

    //move backwards if no wall behind you
    if(sf::Keyboard::isKeyPressed(sf::Keyboard::S))
    {
        if(getMapTile(m_camposx - m_dirx * movespeed, m_camposy) == 0)
            m_camposx -= m_dirx * movespeed;

        if(getMapTile(m_camposx, m_camposy - m_diry * movespeed) == 0)
            m_camposy -= m_diry * movespeed;
    }

    //rotate to the right
    if(sf::Keyboard::isKeyPressed(sf::Keyboard::D))
    {
        //both camera direction and camera plane must be rotated
        const float olddirx = m_dirx;
        m_dirx = m_dirx * std::cos(-rotspeed) - m_diry * sin(-rotspeed);
        m_diry = olddirx * std::sin(-rotspeed) + m_diry * cos(-rotspeed);
        const float oldplanex = m_planex;
        m_planex = m_planex * cos(-rotspeed) - m_planey * std::sin(-rotspeed);
        m_planey = oldplanex * sin(-rotspeed) + m_planey * std::cos(-rotspeed);
    }

    //rotate to the left
    if(sf::Keyboard::isKeyPressed(sf::Keyboard::A))
    {
        //both camera direction and camera plane must be rotated
        float oldDirX = m_dirx;
        m_dirx = m_dirx * std::cos(rotspeed) - m_diry * std::sin(rotspeed);
        m_diry = oldDirX * std::sin(rotspeed) + m_diry * std::cos(rotspeed);
        const float oldplanex = m_planex;
        m_planex = m_planex * std::cos(rotspeed) - m_planey * sin(rotspeed);
        m_planey = oldplanex * std::sin(rotspeed) + m_planey * cos(rotspeed);
    }
}

void CudaRaycaster::setTexture(unsigned texnum, const sf::Image& img)
{
    if(img.getSize() != sf::Vector2u(kTextureSize, kTextureSize))
        return;

    unsigned tex[kTexturePixels];
    for(int x = 0; x < kTextureSize; ++x)
        for(int y = 0; y < kTextureSize; ++y)
            tex[texturePixelIndex(x, y)] = img.getPixel(x, y).toInteger();

    if((texnum * kTexturePixels) >= m_cuda_textures.size())
        m_cuda_textures.resize((texnum + 1u) * kTexturePixels);

    checkCudaCall(cudaMemcpy(m_cuda_textures.ptr() + texnum * kTexturePixels, tex, sizeof(unsigned) * kTexturePixels, cudaMemcpyHostToDevice));
}

void CudaRaycaster::setScreenSize(unsigned width, unsigned height)
{
    if(m_screenwidth == width && m_screenheight == height)
        return;

    height = height - (height % 2); //only even height allowed
    m_screenwidth = width;
    m_screenheight = height;
    m_screenpixels = width * height;
    m_screen.assign(m_screenpixels, 0x7f7f7fff);

    m_cuda_screen.resize(m_screenpixels);
}

void CudaRaycaster::setMapSize(unsigned width, unsigned height)
{
    if(width < 3u || height < 3u)
        return;

    m_mapwidth = width;
    m_mapheight = height;
    m_map.assign(width * height, 0u);

    for(unsigned x = 0; x < width; ++x)
    {
        setMapTile(x, 0u, 1u);
        setMapTile(x, height - 1u, 1u);
    }

    for(unsigned y = 0; y < height; ++y)
    {
        setMapTile(0u, y, 1u);
        setMapTile(width - 1u, y, 1u);
    }
}

void CudaRaycaster::setMapTile(unsigned x, unsigned y, unsigned tile)
{
    if(x < m_mapwidth && y < m_mapheight)
        m_map[x + y * m_mapwidth] = tile;
}

void CudaRaycaster::downloadImage(sf::Texture& texture)
{
    if(texture.getSize() != sf::Vector2u(m_screenwidth, m_screenheight))
        texture.create(m_screenwidth, m_screenheight);

    texture.update(reinterpret_cast<sf::Uint8*>(m_screen.data()));
}

void CudaRaycaster::loadMap(const sf::Image& img)
{
    std::vector<unsigned> tiles;
    const auto ims = img.getSize();
    m_mapwidth = ims.x;
    m_mapheight = ims.y;
    for(unsigned y = 0u; y < ims.y; ++y)
        for(unsigned x = 0u; x < ims.x; ++x)
            tiles.push_back(img.getPixel(x, y) != sf::Color::Black);

    m_map = tiles;
    m_cuda_map.resize(tiles.size());
    checkCudaCall(cudaMemcpy(m_cuda_map.ptr(), tiles.data(), sizeof(tiles[0]) * tiles.size(), cudaMemcpyHostToDevice));
}

CameraExchangeInfo CudaRaycaster::getCameraInfo() const
{
    CameraExchangeInfo ret;
    ret.camposx = m_camposx;
    ret.camposy = m_camposy;
    ret.dirx = m_dirx;
    ret.diry = m_diry;
    ret.planex = m_planex;
    ret.planey = m_planey;
    return ret;
}

void CudaRaycaster::setCameraInfo(const CameraExchangeInfo& info)
{
    m_camposx = info.camposx;
    m_camposy = info.camposy;
    m_dirx = info.dirx;
    m_diry = info.diry;
    m_planex = info.planex;
    m_planey = info.planey;
}

unsigned CudaRaycaster::getMapTile(unsigned x, unsigned y) const
{
    if(x < m_mapwidth && y < m_mapheight)
        return m_map[x + m_mapwidth * y];

    return 1u;
}
