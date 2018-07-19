#pragma once
#include <SFML/System/Vector2.hpp>
#include <cstdio>

const unsigned kResolutionsCount = 5u;

const sf::Vector2u kResolutions[kResolutionsCount] = {
{ 320u, 240u },
{ 640u, 480u },
{ 800u, 600u },
{ 1024u, 768u },
{ 1280u, 1024u },
};

static bool getResolutionText(void * data, int idx, const char ** out)
{
    char * buff = static_cast<char*>(data);
    (void)data;
    if(!(idx < kResolutionsCount))
        return false;

    std::snprintf(buff, 30u, "%ux%u", kResolutions[idx].x, kResolutions[idx].y);
    *out = buff;
    return true;
}

class RunSettings
{
public:
    sf::Vector2u getResolution() const
    {
        return kResolutions[resolution];
    }

    int resolution = 0;
    bool stretch = false;
    bool smooth = false;
    bool fpslock = false;
    bool rasteronly = false;

};
