#include "BackendManager.hpp"
#include <SFML/Graphics/Image.hpp>
#include "RaycasterInterface.hpp"

BackendManager::~BackendManager()
{
    for(RaycasterInterface * ptr : m_backends)
        delete ptr;
}

void BackendManager::loadResources()
{
    for(auto backend : m_backends)
    {
        sf::Image img;
        if(img.loadFromFile("tex1.png"))
            backend->setTexture(1u, img);

        if(img.loadFromFile("tex2.png"))
            backend->setTexture(2u, img);

        if(img.loadFromFile("map.png"))
            backend->loadMap(img);
    }
}

RaycasterInterface * BackendManager::getCurrentInterface() const
{
    if(m_backends.empty())
        return 0x0;

    return m_backends[m_curbackendindex];
}

void BackendManager::switchInterface()
{
    if(m_backends.size() < 2u)
        return;

    m_curbackendindex = (m_curbackendindex + 1u) % m_backends.size();
}
