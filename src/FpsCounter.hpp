#pragma once
#include <SFML/System/Clock.hpp>

class FpsCounter
{
public:
    float frame()
    {
        ++m_frames;
        if(m_frames > 1000u || m_clock.getElapsedTime().asSeconds() > 0.5f)
        {
            m_fps = m_frames / m_clock.getElapsedTime().asSeconds();
            m_frames = 0u;
            m_clock.restart();
        }
        return m_fps;
    }

private:
    sf::Clock m_clock;
    unsigned m_frames = 0u;
    float m_fps = 0.f;

};
