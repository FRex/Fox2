#include "BackendManager.hpp"

BackendManager::~BackendManager()
{
    for(RaycasterInterface * ptr : m_backends)
        delete ptr;
}

void BackendManager::loadResources()
{

}

RaycasterInterface * BackendManager::getCurrentInterface() const
{
    if(m_backends.empty())
        return 0x0;

    return m_backends[m_curbackendindex];
}
