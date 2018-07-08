#pragma once
#include <vector>
#include <SFML/System/NonCopyable.hpp>

class RaycasterInterface;
class RunInfo;

class BackendManager : sf::NonCopyable
{
public:
    ~BackendManager();
    template <class T> T * addBackend();

    void loadResources();
    RaycasterInterface * getCurrentInterface() const;
    void switchInterface();

private:
    std::vector<RaycasterInterface*> m_backends;
    unsigned m_curbackendindex = 0u;

};

template<class T>
inline T * BackendManager::addBackend()
{
    const auto ret = new T;
    m_backends.push_back(ret);
    return ret;
}
