#pragma once
#include <sstream>

const unsigned kResolutionsCount = 5u;

const sf::Vector2u kResolutions[kResolutionsCount] = {
{ 320u, 240u },
{ 640u, 480u },
{ 800u, 600u },
{ 1024u, 768u },
{ 1280u, 1024u },
};

class RunInfo
{
public:
    std::string toString() const
    {
        std::ostringstream ss;
        ss << std::boolalpha;
        ss << "FPS:   " << fps << std::endl;
        ss << "RTime: " << rastertime << std::endl;
        ss << "(R) Stretch:    " << stretch << std::endl;
        ss << "(T) Smooth:     " << smooth << std::endl;
        ss << "(U) 60FPS lock: " << fpslock << std::endl;
        ss << "(Y) Resolution: " << kResolutions[resolution].x << 'x' << kResolutions[resolution].y << std::endl;
        ss << "(I) Type:       " << rendertype << std::endl;
        ss << "(O) Rasteronly: " << rasteronly << std::endl;
        return ss.str();
    }

    sf::Vector2u getResolution() const
    {
        return kResolutions[resolution];
    }

    float fps = 0.f;
    bool stretch = false;
    bool smooth = false;
    unsigned resolution = 0u;
    const char * rendertype = "";
    bool fpslock = false;
    float rastertime = 0.f;
    bool rasteronly = false;

};
