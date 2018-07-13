#include "FoxRaycaster.hpp"
#include <SFML/Graphics.hpp>
#include <sstream>
#include <algorithm>
#include "BackendManager.hpp"
#include "FpsCounter.hpp"
#include "RunInfo.hpp"
#include "CudaRaycaster.hpp"


int main(int argc, char ** argv)
{
    FpsCounter fps;
    RunInfo runinfo;

    sf::RenderWindow app(sf::VideoMode(800u, 600u), "FoxRaycaster");
    app.setFramerateLimit(60u);

    BackendManager manager;
    manager.addBackend<fox::FoxRaycaster>();
    manager.addBackend<fox::CudaRaycaster>();
    manager.loadResources();

    RaycasterInterface * currentraycaster = manager.getCurrentInterface();
    currentraycaster->setScreenSize(runinfo.getResolution().x, runinfo.getResolution().y);

    sf::Texture tex;
    sf::Font font;
    font.loadFromFile("DejaVuSansMono.ttf");
    while(app.isOpen())
    {
        sf::Event eve;
        while(app.pollEvent(eve))
        {
            switch(eve.type)
            {
            case sf::Event::Closed:
                app.close();
                break;
            case sf::Event::Resized:
                app.setView(sf::View(sf::FloatRect(0.f, 0.f, eve.size.width, eve.size.height)));
                break;
            case sf::Event::KeyPressed:
                switch(eve.key.code)
                {
                case sf::Keyboard::R:
                    runinfo.stretch = !runinfo.stretch;
                    break;
                case sf::Keyboard::T:
                    runinfo.smooth = !runinfo.smooth;
                    break;
                case sf::Keyboard::Y:
                    runinfo.resolution = (runinfo.resolution + 1) % kResolutionsCount;
                    currentraycaster->setScreenSize(runinfo.getResolution().x, runinfo.getResolution().y);
                    break;
                case sf::Keyboard::U:
                    runinfo.depthdraw = !runinfo.depthdraw;
                    break;
                case sf::Keyboard::I:
                    manager.switchInterface();
                    currentraycaster = manager.getCurrentInterface();
                    currentraycaster->setScreenSize(runinfo.getResolution().x, runinfo.getResolution().y);
                    break;
                }//switch eve key code
                break;
            }
        }//while app poll event eve

        app.clear(sf::Color(0x2d0022ff));
        currentraycaster->handleKeys();
        currentraycaster->rasterize();
        if(runinfo.depthdraw)
        {
            currentraycaster->downloadDepthImage(tex);
        }
        else
        {
            currentraycaster->downloadImage(tex);
        }

        sf::Sprite spr(tex);
        tex.setSmooth(runinfo.smooth);
        if(runinfo.stretch)
        {
            const auto ts = sf::Vector2f(tex.getSize());
            const auto ws = sf::Vector2f(app.getSize());
            const float s = std::min(ws.x / ts.x, ws.y / ts.y);
            spr.setScale(s, s);
        }//if runinfo stretch

        spr.setOrigin(sf::Vector2f(tex.getSize()) / 2.f);
        spr.setPosition(sf::Vector2f(app.getSize()) / 2.f);
        app.draw(spr);

        runinfo.rendertype = currentraycaster->getRaycasterTechName();
        sf::Text txt(runinfo.toString(), font);
        txt.setOutlineThickness(1.f);
        txt.setFillColor(sf::Color::White);
        txt.setOutlineColor(sf::Color::Black);
        app.draw(txt);
        app.display();
        runinfo.fps = fps.frame();
    }
    return 0;
}
