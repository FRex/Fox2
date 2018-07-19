#include "AppFox2.hpp"
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include "FoxRaycaster.hpp"
#include "CudaRaycaster.hpp"
#include "single_imgui.hpp"

void AppFox2::run()
{
    init();
    while(m_win.isOpen())
    {
        update();
        gui();
        draw();
    }
    shutdown();
}

void AppFox2::init()
{
    m_win.create(sf::VideoMode(640u, 480u), "Fox2");
    ImGui::SFML::Init(m_win);
    m_manager.addBackend<fox::FoxRaycaster>();
    m_cudaraycaster = m_manager.addBackend<CudaRaycaster>();
    m_manager.loadResources();
    auto cr = m_manager.getCurrentInterface();
    cr->setScreenSize(640u, 480u);
}

void AppFox2::shutdown()
{
    ImGui::SFML::Shutdown();
}

void AppFox2::update()
{
    sf::Event eve;
    while(m_win.pollEvent(eve))
    {
        ImGui::SFML::ProcessEvent(eve);
        switch(eve.type)
        {
        case sf::Event::Closed:
            m_win.close();
            break;
        case sf::Event::Resized:
            m_win.setView(sf::View(sf::FloatRect(0.f, 0.f, eve.size.width, eve.size.height)));
            break;
        }//switch eve type
    }//while
    m_manager.getCurrentInterface()->handleKeys();
    ImGui::SFML::Update(m_win, m_guiclock.restart());

}

void AppFox2::draw()
{
    m_win.setFramerateLimit(60u * m_runsettings.fpslock);
    m_win.clear(sf::Color(0x2d0022ff));
    auto cr = m_manager.getCurrentInterface();
    cr->setScreenSize(m_runsettings.getResolution().x, m_runsettings.getResolution().y);
    cr->rasterize();
    if(!m_runsettings.rasteronly)
    {
        cr->downloadImage(m_texture);
        m_texture.setSmooth(m_runsettings.smooth);
        sf::Sprite spr(m_texture);
        if(m_runsettings.stretch)
        {
            const auto ts = sf::Vector2f(m_texture.getSize());
            const auto ws = sf::Vector2f(m_win.getSize());
            const float s = std::min(ws.x / ts.x, ws.y / ts.y);
            spr.setScale(s, s);
        }//if stretch
        spr.setOrigin(sf::Vector2f(m_texture.getSize()) / 2.f);
        spr.setPosition(sf::Vector2f(m_win.getSize()) / 2.f);
        m_win.draw(spr);
    }//if not rasteronly
    ImGui::SFML::Render(m_win);
    m_win.display();
}

void AppFox2::gui()
{
    ImGui::Begin("Fox2");
    ImGui::Text("Interface");
    ImGui::SameLine();
    if(ImGui::Button(m_manager.getCurrentInterface()->getRaycasterTechName()))
        m_manager.switchInterface();

    char buff[128];
    ImGui::Combo("Resolution", &m_runsettings.resolution, getResolutionText, buff, kResolutionsCount, kResolutionsCount + 1);
    ImGui::Text("FPS: %f\n", m_fpscounter.frame());
    ImGui::Checkbox("Stretch", &m_runsettings.stretch);
    ImGui::Checkbox("Smooth", &m_runsettings.smooth);
    ImGui::Checkbox("60FPS Lock", &m_runsettings.fpslock);
    ImGui::Checkbox("Raster only", &m_runsettings.rasteronly);
    ImGui::End();
}