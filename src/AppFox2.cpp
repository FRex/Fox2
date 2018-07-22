#include "AppFox2.hpp"
#include <SFML/Window/Event.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include "FoxRaycaster.hpp"
#include "CudaRaycaster.hpp"
#include "single_imgui.hpp"
#include "gl_core21.h"

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
    if(ogl_LoadFunctions() == ogl_LOAD_FAILED)
    {
        m_win.close();
        return;
    }
    m_glvendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
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
    if(m_movementclock.getElapsedTime() > sf::seconds(1.f / 60.f))
    {
        m_manager.getCurrentInterface()->handleKeys();
        m_movementclock.restart();
    }
    ImGui::SFML::Update(m_win, m_guiclock.restart());

}

void AppFox2::draw()
{
    m_win.setFramerateLimit(m_runsettings.fpslock * 60u);
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

    ImGui::Text("GL_VENDER = %s", m_glvendor.c_str());
    char buff[128];
    ImGui::Combo("Resolution", &m_runsettings.resolution, getResolutionText, buff, kResolutionsCount, kResolutionsCount + 1);
    ImGui::Text("FPS: %f\n", m_fpscounter.frame());
    ImGui::Checkbox("Stretch", &m_runsettings.stretch);
    ImGui::Checkbox("Smooth", &m_runsettings.smooth);
    ImGui::Checkbox("Raster only", &m_runsettings.rasteronly);
    ImGui::Checkbox("60 FPS Lock", &m_runsettings.fpslock);
    ImGui::Separator();
    ImGui::Text("raster time: %fms", m_manager.getCurrentInterface()->getRasterTime());
    if(m_cudaraycaster == m_manager.getCurrentInterface())
    {
        int tpb = m_cudaraycaster->getThreadsPerBlock();
        if(ImGui::InputInt("TPB", &tpb))
            m_cudaraycaster->setThreadsPerBlock(tpb);
    }//if
    ImGui::End();
}
