#pragma once
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>
#include "BackendManager.hpp"
#include "RunSettings.hpp"
#include "FpsCounter.hpp"

class CudaRaycaster;

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
    RunSettings m_runsettings;
    CudaRaycaster * m_cudaraycaster = 0x0;
    sf::Clock m_movementclock;
    std::string m_glvendor;
    std::string m_glrenderer;
    int m_framecounter = 0;

};
