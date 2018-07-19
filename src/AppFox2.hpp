#pragma once
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>
#include "BackendManager.hpp"
#include "RunInfo.hpp"
#include "FpsCounter.hpp"

class AppFox2 : sf::NonCopyable
{
public:
    void run();

private:
    void init();
    void shutdown();
    void update();
    void draw();
    void gui();

    sf::RenderWindow m_win;
    BackendManager m_manager;
    sf::Clock m_guiclock;
    sf::Texture m_texture;
    FpsCounter m_fpscounter;

};
